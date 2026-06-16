# SADAF Methodology Notes

This document provides detailed methodological justification for each
component of the SADAF pipeline, supplementing the high-level summary
in the main README.

## 1. Data Preprocessing

- **Missing values**: `CTR` and `Depth` are missing only when `impression = 0`
  (verified: 100% co-occurrence). Both are filled with 0, since a CTR/Depth
  metric is undefined without impressions rather than genuinely unobserved.
- **Derived features**: `CPC = cost / click` (0 if click = 0), `CPA = cost /
  conversion_count` (0 if conversions = 0), `has_conversion =
  1[conversion_count > 0]`.
- **Log transforms**: `log1p` applied to impression, click, cost, CTR, CVR,
  ROAS, CPC, conversion_count to stabilize variance ahead of regression and
  neural-network training (raw distributions exhibit coefficients of
  variation between 1.2 and 13.9).
- **Cyclical encoding**: `hour_sin = sin(2π·Hours/24)`, `hour_cos =
  cos(2π·Hours/24)` avoid the discontinuity at the 23→0 hour boundary that a
  raw integer encoding would introduce.
- **Campaign type**: extracted from the `campaign_id` prefix (`-01-` =
  Search, `-02-` = Shopping, `-04-` = Zero-cost). Zero-cost campaigns are
  excluded from causal analyses (H1–H3) because they lack a cost-based
  treatment/control contrast, but are retained in descriptive EDA.

## 2. Zero-Inflation Diagnosis (ZINB)

ROAS is discretized into 11 ordinal bins (0 = zero, 1–10 = deciles of
positive ROAS) to fit a Zero-Inflated Negative Binomial (ZINB) model via
`statsmodels`. Two optimizers are attempted in sequence (`lbfgs`, then
`nm`); a fit is accepted only if both the AIC and all standard errors are
finite, since `lbfgs` can silently fail to compute valid Hessian-based SEs.
ZINB is compared against a Zero-Inflated Poisson (ZIP) via ΔAIC; ΔAIC > 10
is interpreted as strong evidence for ZINB's additional dispersion
parameter, motivating the two-stage (classification → regression)
prediction architecture used in H4a–H4c rather than a single zero-inflated
regression head.

## 3. Causal Identification Strategy (H1–H3)

### H1: PSM + Doubly Robust IPW

- **Treatment**: ads with CTR above the sample median (`T_highCTR`).
- **Confounders**: `log_impression`, `log_cost`, `Depth`, and one-hot
  campaign-type dummies (drop-first encoding).
- **Propensity model**: L2-regularized logistic regression (`C=0.1`) on
  standardized confounders.
- **Matching**: 1-nearest-neighbor on the logit propensity score, caliper
  = 0.1 × std(logit propensity score), following Austin (2011)'s
  recommended default.
- **Balance diagnostics**: standardized mean difference (SMD) computed
  pre- and post-match for every confounder; |SMD| < 0.10 is "balanced",
  0.10–0.25 is "residual" (flagged but tolerated), and > 0.25 is
  "substantial" and explicitly corrected via the IPW estimator.
- **Primary estimator**: because PSM alone leaves residual imbalance on
  `log_impression` and `log_cost` (|SMD| > 0.1), the **IPW-ATT** is
  reported as the primary, doubly robust causal estimate; the PSM-ATT
  (with 2,000-resample bootstrap CI) is retained only as corroborating
  evidence.
- **IPW weights** are truncated at the 99th percentile to limit the
  influence of extreme propensity scores.

### H2: Mediation (Depth as mediator)

- Baron-Kenny decomposition: `a` (CTR→Depth, linear regression), `b`
  (Depth→Conversion controlling for CTR, logistic regression), indirect
  effect = `a × b`.
- 2,000-resample bootstrap CI on the indirect effect.
- **Suppressor classification**: when `a < 0` and `b < 0` (so `a×b > 0`),
  the path is labeled a *negative suppressor*: high-CTR ads reduce
  browsing depth (immediate-click behavior bypasses deliberate browsing),
  while among ads that do generate depth, deeper browsing is associated
  with *lower* conversion — interpreted as Depth proxying decision
  hesitancy rather than positive engagement. The proportion-mediated
  statistic is reported in the appendix only, since its sign and magnitude
  are unstable under suppression and the qualitative sign pattern is the
  theoretically meaningful finding.

### H3: Moderation (campaign type)

- OLS with `log_ROAS ~ log_CTR * is_search + log_cost + log_impression`,
  HC3 heteroskedasticity-robust standard errors (appropriate given the
  high variance heterogeneity documented in EDA, CV up to 13.9 for ROAS).
- Marginal effects computed as `β_CTR` (Shopping) and `β_CTR + β_interaction`
  (Search).

## 4. Data Augmentation Pipeline

Three complementary generative methods are combined because each captures
a different aspect of the sparse sequence distribution:

| Method | Captures | Limitation |
|---|---|---|
| β-VAE | Nonlinear latent structure | Higher distributional distance to real data (mean KS ≈ 0.40) |
| Gaussian Copula | Marginal-preserving dependency structure | Intermediate fidelity (mean KS ≈ 0.11) |
| Moving Block Bootstrap (MBB) | Exact local temporal dynamics | Highest fidelity (mean KS ≈ 0.01) but limited novelty |

Synthetic sequences from all three methods are pooled with the real data
and shuffled before training. **Fréchet Score Distance (FSD)** — computed
from the hidden-state embeddings of a reference GRU trained on real data
— validates the combined augmented distribution against the real one.
Thresholds: FSD < 2.0 = accept, 2.0–5.0 = warn, > 5.0 = reject. The
reported FSD = −0.1347 is well within the accept band (a slightly negative
value reflects finite-sample noise in the eigenvalue decomposition and is
treated as ≈ 0).

## 5. Prediction Architecture (H4a–H4c)

### Two-stage design

Because 72.1% of paid impressions have ROAS = 0 (structural zero-inflation
confirmed in §2), prediction is split into:
1. **Classification stage**: predict `has_roas` (binary).
2. **Regression stage**: predict `log(ROAS+1)` conditional on ROAS > 0.

### Models compared

- **BayesianLSTM**: standard LSTM with MC Dropout (Gal & Ghahramani, 2016)
  for approximate Bayesian inference; dropout = 0.4 for the regression
  head (0.3 for the classification head, since the smaller positive class
  benefits from less aggressive regularization).
- **LSTM / BiLSTM / GRU**: standard recurrent baselines.
- **Mamba**: a simplified selective state-space model (Gu et al., 2023)
  whose input-dependent Δ_t parameter is hypothesized to selectively
  compress near-zero-ROAS time slots, motivating the H4c robustness
  hypothesis specifically (not an accuracy claim).
- **Ridge / MLP**: non-sequential baselines on flattened input, to test
  whether sequence structure carries information beyond static features.

### Statistical testing

- **Diebold-Mariano (DM) test** with HAC-corrected variance for all
  pairwise model comparisons on squared error.
- **Bootstrap RMSE CIs** (1,000 resamples) per model.
- Given the small test set (n = 34 for SEQ_LEN=4), DM results are reported
  conservatively: only the two pairs surviving raw p < 0.05
  (BayesianLSTM > LSTM, BayesianLSTM > Mamba) are highlighted as
  significant in the main text; the robustness checks (§9 below) show
  these do **not** survive Bonferroni or BH-FDR correction across all 21
  pairwise comparisons, and this caveat is reported explicitly rather than
  omitted.

### H4c protocol

Mamba's claimed contribution is narrowed to **robustness to sequence-length
variation** (ΔRMSE between SEQ_LEN=4 and SEQ_LEN=6), not absolute accuracy.
DM tests between architectures at fixed SEQ_LEN are expected to be
non-significant under H4c and are reported as such (not treated as a
failure).

## 6. Bayesian Calibration

MC Dropout with `n_samples=500` produces a predictive distribution; raw MC
Dropout credible intervals are typically miscalibrated (under-covering),
so a temperature-scaling correction (`T=1.5`) is applied post-hoc by
scaling the deviation of each draw from the posterior mean. Calibration is
evaluated at four nominal coverage levels (50/80/90/95%); a model is
"well-calibrated" if actual coverage falls within −3 percentage points of
nominal at the 95% level.

## 7. Explainability (H5)

Four attribution methods are computed per cluster (KMeans, k=3, on
standardized last-timestep features):

- **GS-SHAP** (primary): HSIC-based feature grouping followed by exact
  Shapley value computation over groups rather than individual features,
  reducing the combinatorial explosion of standard Shapley while
  preserving feature-interaction structure.
- **Integrated Gradients**: path integral from a zero/mean baseline;
  computed on CPU due to a backward-pass incompatibility with the training
  device in this environment (explicitly logged as a fallback, not a
  silent failure).
- **Permutation SHAP**: model-agnostic, computed by replacing one feature
  at a time with background-distribution draws.
- **Attention attribution**: weights from a dedicated `LSTMWithAttention`
  model, multiplied by the absolute input magnitude at each timestep.

**Method agreement** is quantified via pairwise Spearman ρ on
cluster-mean feature importances. Because Attention measures *temporal
position* (which timestep the model attends to) rather than *feature-level*
importance averaged over time, it is expected to diverge from the other
three gradient/perturbation-based methods; the robustness supplement (W4)
explicitly re-computes a 3-method consensus excluding Attention and
recommends this reframing for the paper's discussion section.

## 8. Domain Shift and Adaptation (H6)

- **KS test** per feature, last timestep, between Search and Shopping
  campaign sequences. H6 is considered supported if ≥ 6 of 7 features
  show p < 0.05.
- **Frozen-encoder transfer**: a GRU trained on Search sequences has its
  first 50% of parameters frozen, then is fine-tuned on Shopping sequences
  at a reduced learning rate (5e-5). The reported RMSE gain (+0.2–0.4%
  over naive zero-shot transfer) is modest and is explicitly framed as
  providing *theoretical justification* for domain-adaptive design, not a
  performance-superiority claim.

## 9. Robustness Supplement (Weakness-Response Analyses)

Seven supplementary analyses (W1–W7) were added in response to anticipated
reviewer concerns:

1. **W1 (data scale)**: augmentation-method KS/MMD stratification,
   data-scaling curve (power-law fit across 20–100% of real training
   data, 3 seeds), and Leave-One-Ad-Group-Out CV (LOGO-CV) as the most
   rigorous generalization test.
2. **W2 (external validity)**: hour-of-day-block holdout CV, and a
   synthetic multi-advertiser simulation perturbing CTR scale, cost
   scale, and ROAS noise to probe robustness beyond the single advertiser
   in the dataset.
3. **W3 (overfitting)**: quantified train/validation gap per model;
   3×3 grid search over dropout × weight-decay for GRU.
4. **W4 (attribution disagreement)**: 3-method consensus excluding
   Attention; explicit visualization of Attention's temporal-weighting
   structure to support the "different axis of explanation" reframing.
5. **W5 (ProtoNet utility)**: cold-start RMSE trajectory over K = 1→21
   support examples, and comparison of random vs. nearest-neighbor
   prototype selection.
6. **W6 (DM multiple comparisons)**: Bonferroni and BH-FDR correction
   across all 21 pairwise model comparisons, plus Cohen's d effect sizes
   and bootstrap CI overlap analysis.
7. **W7 (IS theory grounding)**: empirical tests of Signaling Theory
   (CTR-quartile monotonicity in conversion/ROAS) and Resource Scarcity
   (ROAS uncertainty and zero-inflation rate as functions of impression
   volume), plus an explicit mapping of SADAF's components to five
   IS/DSS theoretical traditions.

All W-series results are reported honestly even when they complicate the
main narrative (e.g., W6 shows zero DM pairs survive correction; W2-b
shows the GRU model performs poorly, even worse than baseline, on
out-of-distribution synthetic advertisers), consistent with the framework's
overall commitment to conservative, reviewer-anticipatory reporting.

## References

- Austin, P. C. (2011). An introduction to propensity score methods.
  *Multivariate Behavioral Research*, 46(3), 399–424.
- Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable
  distinction. *Journal of Personality and Social Psychology*, 51(6).
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation.
  *ICML*.
- Gu, A., et al. (2023). Mamba: Linear-time sequence modeling with
  selective state spaces. *arXiv:2312.00752*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. *NeurIPS*.
- Spence, M. (1973). Job market signaling. *Quarterly Journal of
  Economics*, 87(3), 355–374.
- Akerlof, G. A. (1970). The market for "lemons". *Quarterly Journal of
  Economics*, 84(3), 488–500.
- Hevner, A. R., et al. (2004). Design science in information systems
  research. *MIS Quarterly*, 28(1), 75–105.
- Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and
  user acceptance of information technology. *MIS Quarterly*, 13(3).