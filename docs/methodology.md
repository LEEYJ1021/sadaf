# SADAF Methodology Notes

This document provides detailed methodological justification for each
component of the SADAF pipeline, supplementing the high-level summary
in the main README.

> **v6.0 alignment note.** This file has been revised to match the
> submitted manuscript and the current `README.md` (v6.0). Two
> structural corrections carry through every section below:
> (1) the dataset is a **37-advertiser panel**, not a single advertiser
> — every sample size stated here (matched pairs, paid rows, test
> sequences, cluster sizes) is pooled or drawn across all 37 advertisers
> unless stated otherwise; and (2) the data is standardized and provided
> by **SearchM Co., Ltd.**, an official agency unifying **Naver and
> Kakao** search-advertising operations — it is not Naver-only. Where an
> earlier draft's numbers or model rankings differ from the ones below
> (e.g., an older H4a/H4b run with a smaller test split), this file
> reports only the figures that appear in the submitted manuscript and
> `README.md`; superseded figures are not retained here to avoid
> confusion (see `readme/README_v5_full.md` for the pre-redesign
> pipeline's captured stdout, kept for provenance only).

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
  variation between 1.2 and 13.9, pooled across the 37-advertiser panel).
- **Cyclical encoding**: `hour_sin = sin(2π·Hours/24)`, `hour_cos =
  cos(2π·Hours/24)` avoid the discontinuity at the 23→0 hour boundary that a
  raw integer encoding would introduce.
- **Campaign type**: extracted from the `campaign_id` prefix (`-01-` =
  Search, `-02-` = Shopping, `-04-` = Zero-cost). Zero-cost campaigns are
  excluded from causal analyses (H1–H3) because they lack a cost-based
  treatment/control contrast, but are retained in descriptive EDA.
- **Advertiser identifier**: every row carries an anonymized advertiser
  identifier in addition to the ad-group identifier. This is the field
  leave-one-advertiser-out cross-validation (§9, RQ6) partitions on, and
  the field that distinguishes "37 advertisers" from "37 ad groups" — a
  single advertiser typically contributes multiple ad groups, so the two
  counts are not interchangeable (see `figures/cluster_sizes.csv` for the
  unique-ad-group-vs-row-level distinction used in §7).

## 2. Zero-Inflation Diagnosis (ZINB)

ROAS is discretized into 11 ordinal bins (0 = zero, 1–10 = deciles of
positive ROAS) to fit a Zero-Inflated Negative Binomial (ZINB) model via
`statsmodels`, pooled across the full 37-advertiser panel (n = 32,494 paid
rows). Two optimizers are attempted in sequence (`lbfgs`, then `nm`); a fit
is accepted only if both the AIC and all standard errors are finite, since
`lbfgs` can silently fail to compute valid Hessian-based SEs. ZINB is
compared against a Zero-Inflated Poisson (ZIP) via ΔAIC; the observed
ΔAIC = −2,798.9 is interpreted as strong evidence for ZINB's additional
dispersion parameter, motivating the two-stage (classification →
regression) prediction architecture used in H4a–H4c rather than a single
zero-inflated regression head. Given this row-level sample size, the
ZINB-vs-ZIP comparison itself is well-powered; the power constraints
discussed in the main text's power section apply to the sequence-level
(H4a–c) and cluster-level (H5) tests only.

## 3. Causal Identification Strategy (H1–H3)

All causal-pillar estimates below are computed on samples **pooled across
the full 37-advertiser panel** — this is a deliberate design choice
(Section 5.1 of the manuscript) that gives the causal pillar the largest,
best-powered sample of any pillar in SADAF.

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
  `log_impression` and `log_cost` (|SMD| > 0.1), the **doubly robust
  IPW-ATT (= 0.1286)** is reported as the primary causal estimate; the
  PSM-ATT (= 0.1347, 95% CI [0.1254, 0.1434], n_matched = 14,987) is
  retained as corroborating evidence. The two estimators agree within
  0.006 — below the pre-specified 0.05 consistency threshold — precisely
  because the doubly robust correction absorbs the residual imbalance
  that PSM alone leaves uncorrected.
- **IPW weights** are truncated at the 99th percentile to limit the
  influence of extreme propensity scores.

### H2: Mediation → Suppression (Depth)

- Baron-Kenny decomposition: `a` (CTR→Depth, linear regression), `b`
  (Depth→Conversion controlling for CTR, logistic regression), indirect
  effect = `a × b`.
- 2,000-resample bootstrap CI on the indirect effect: observed
  `a = −0.3077`, `b = −0.0861`, indirect = `0.0265`, 95% CI [0.0200, 0.0337].
- **Suppressor classification**: because `a < 0` and `b < 0` (so
  `a×b > 0`), the path is labeled a *negative suppressor* rather than
  mediation: high-CTR ads reduce browsing depth (immediate-click behavior
  bypasses deliberate browsing), while among ads that do generate depth,
  deeper browsing is associated with *lower* conversion — interpreted as
  Depth proxying decision hesitancy rather than positive engagement.
- **Reporting discipline**: H2 was originally specified as a mediation
  hypothesis (Depth transmitting part of CTR's effect on conversion in a
  consistent direction). The observed pattern is the opposite structural
  signature. This is reported throughout the manuscript and README as an
  **informative departure from the hypothesized mechanism**, not as
  "H2 supported" — labeling it "supported" without qualification would
  misrepresent what was actually found. The indirect effect is still
  statistically distinguishable from zero and is retained as a
  substantively meaningful finding in its own right.

### H3: Moderation (campaign type)

- OLS with `log_ROAS ~ log_CTR * is_search + log_cost + log_impression`,
  HC3 heteroskedasticity-robust standard errors (appropriate given the
  high variance heterogeneity documented in EDA, CV up to 13.9 for ROAS),
  n = 9,069 (pooled across the 37-advertiser panel), R² = 0.655.
- Observed: `β_interaction = 0.386` (p < .0001); marginal effect,
  Search = 0.949; marginal effect, Shopping = 0.563.
- Marginal effects computed as `β_CTR` (Shopping) and `β_CTR + β_interaction`
  (Search).

## 4. Data Augmentation Pipeline

Three complementary generative methods are combined because each captures
a different aspect of the sparse sequence distribution:

| Method | Captures | Limitation |
|---|---|---|
| β-VAE | Nonlinear latent structure | Higher distributional distance to real data |
| Gaussian Copula | Marginal-preserving dependency structure | Intermediate fidelity |
| Moving Block Bootstrap (MBB) | Exact local temporal dynamics | Highest fidelity but limited novelty |

Synthetic sequences from all three methods are pooled with the real data
(174 real → ~870 augmented) and shuffled before training. **Fréchet
Sequence Distance (FSD)** — computed from the hidden-state embeddings of
a reference GRU trained on real data — validates the combined augmented
distribution against the real one. Threshold: FSD < 2.0 = accept. The
reported FSD = −0.047 is well within the accept band; because this
implementation is a bias-corrected estimator, a value at or near zero —
including small negatives — indicates the strongest possible pass, not a
violation of non-negativity. Validation and test sequences are never
augmented and remain strictly real throughout. Note: the augmentation
modules (`copula.py`, `mbb.py`) do not yet accept an explicit random seed,
so the augmented training set is not bit-for-bit reproducible across
separate pipeline runs, though the FSD gate consistently passes (see
README §7, open items).

## 5. Prediction Architecture (H4a–H4c)

### Two-stage design

Because 72.1% of paid impressions have ROAS = 0 (structural zero-inflation
confirmed in §2), prediction is split into:
1. **Classification stage (H4a)**: predict `has_roas` (binary), n = 174
   train / 24 validation / 24 test sequences, drawn from the full
   37-advertiser panel.
2. **Regression stage (H4b, H4c)**: predict `log(ROAS+1)` conditional on
   ROAS > 0, same split.

### Models compared

- **Logistic Regression / Ridge Regression**: parsimonious linear
  baselines — the models this pipeline is specifically testing for
  competitiveness against deeper architectures under extreme sparsity
  (H4a).
- **Multilayer Perceptron (MLP)**: non-sequential neural baseline on
  flattened input, to test whether sequence structure carries information
  beyond static features.
- **LSTM / BiLSTM / GRU**: standard recurrent baselines.
- **Bayesian LSTM**: standard LSTM with MC Dropout (Gal & Ghahramani, 2016)
  for approximate Bayesian inference.
- **Mamba**: a selective state-space model (Gu & Dao, 2023) whose
  inclusion is a pre-specified test of whether its long-sequence design
  advantages survive under the four-to-six-step sequences characteristic
  of cold-start ad-group data (H4c, R1) — it is not included on the
  expectation that it will win on headline accuracy.

### H4a result

Logistic regression attains the highest AUC of any classifier (0.6143),
outperforming LSTM (0.6115), Bayesian LSTM (0.5894), and MLP (0.5445). At
174 real training sequences, the binary conversion signal is close to
linearly separable, and the added representational capacity of a
recurrent classifier is actively counterproductive rather than merely
unnecessary — **H4a is supported**, consistent with a classical
bias-variance account of small-sample model selection.

### H4b result and statistical testing

- **Diebold-Mariano (DM) test**, reported both raw and after
  Benjamini-Hochberg false-discovery-rate (FDR) correction, for all
  pairwise model comparisons on squared forecasting error (7 architectures
  → 21 pairs).
- Of 21 pairs, **3 do not converge** (Mamba–Ridge, Mamba–MLP, Ridge–MLP),
  leaving **18 computable pairs**. Of these, **13 are significant at raw
  p < .05**; **8 remain significant after FDR correction**.
- LSTM (RMSE = 1.2099) is the best regression-stage forecaster and
  significantly outperforms GRU, BiLSTM, Mamba, Ridge, MLP, and Bayesian
  LSTM at FDR-corrected significance — **H4b is supported**.
- Given the small test set (n = 24), non-significant pairwise comparisons
  are read via the **a priori minimum-detectable-effect (MDE)** at this
  sample size (dz = 0.60 raw α, dz = 0.88 FDR-equivalent α), not via
  post-hoc power, which is a deterministic function of the already-reported
  p-values and carries no independent corroborating information (Hoenig &
  Heisey, 2001).

### H4c protocol and result

The domain-gap diagnostic pairs each architecture's training loss with the
*same* epoch used to identify its best real-validation loss (strict
epoch-matching), rather than the epoch that minimizes augmented-validation
loss, since the latter would compare quantities from different points in
training. Under this diagnostic, **Mamba is the only architecture showing
the classical overfitting signature** (real-validation loss exceeds
training loss at the matched epoch: 0.7910 vs. 0.6768); the other four
architectures (Bayesian LSTM, LSTM, GRU, BiLSTM) show the opposite
ordering. This categorical difference is reported as a descriptive
diagnostic pattern, not tested for statistical significance, since
`gap_real` is a deterministic function of a single best-epoch loss pair
per architecture rather than a distribution over independent samples.
**H4c is supported** under this revised, descriptive formulation.

As a supplementary robustness check (**R1**, not an independent
hypothesis), Mamba is additionally evaluated at a six-step sequence
length to test whether its weaker four-step accuracy is a length
artifact; it remains the weakest or near-weakest performer at both
lengths, corroborating prior evidence (Liu et al., 2025; Wang et al.,
2025) that selective state-space architectures can be disadvantaged on
short, cold-start-length sequences.

## 6. Bayesian Calibration (exploratory, not part of the core hypothesis set)

MC Dropout with `n_samples=500` produces a predictive distribution for the
Bayesian LSTM; raw MC Dropout credible intervals are typically
miscalibrated (under-covering), so a temperature-scaling correction is
applied post-hoc by scaling the deviation of each draw from the posterior
mean. Calibration is evaluated at four nominal coverage levels
(50/80/90/95%).

> **Scope note.** This calibration analysis, and the Prototypical-Network
> (ProtoNet) cold-start K-shot experiments in `sadaf/models/protonet.py`,
> are retained in the codebase as exploratory analyses that informed the
> pipeline's development. They are **not** part of the core H1–H5 / R1–R2
> / RQ6 hypothesis set reported in the manuscript and `README.md`, and no
> claims from either analysis appear in the submitted paper. They are
> documented here only so that a reader of the repository understands
> what these modules are for; see `results_table.md` for their status.

## 7. Explainability (H5) — individual-level attribution, cross-verified

**This section supersedes an earlier group-level GS-SHAP (HSIC-grouped
Shapley) design.** Group-level Shapley methods are conventionally
motivated by, and most informative in, high-dimensional feature spaces
(Chamma, Thirion & Engemann, 2024); this study's 7-feature set does not
meet that threshold, and the added complexity of a joint group-level
estimator is not empirically warranted here. The current design instead
computes attribution **individually, feature by feature**, using three
independently computed methods on the identical trained regression-stage
model and the identical held-out data, and treats **cross-method
convergence** — not agreement within a single joint estimator — as the
evidentiary standard:

- **Individual SHAP** (primary): kernel-based Shapley-value attribution
  (Lundberg & Lee, 2017), estimated against a 100-sample background
  reference set (subsampled from 200 available background rows).
- **Permutation SHAP**: a Monte Carlo permutation-based Shapley estimator
  (Štrumbelj & Kononenko, 2014), computed via `shap.explainers.Permutation`.
- **Integrated Gradients** (Sundararajan, Taly & Yan, 2017): a gradient-path
  attribution method, accumulating the model's gradient along a path from
  a baseline input to the actual input.

`Attention attribution` (used in an earlier draft) has been **dropped
entirely**, not merely reweighted or excluded from one supplementary
consensus calculation as before. Attention weights measure *temporal
position* (which timestep the model attends to) rather than feature-level
importance averaged over time, so they were never comparable to the other
three methods on the same footing; rather than retaining a fourth,
structurally different method and explaining away its disagreement, the
design now uses only the three methods that estimate the same class of
quantity (feature-level importance), and lets their agreement or
disagreement speak for itself.

**Clustering**: K-means (k=3) on standardized engagement/spend features,
computed at the row level (paid, non-zero-ROAS observations), not the
sequence-level test split. This is a deliberate change from the earlier
design: moving from sequence-level clusters (7–10 test sequences per
cluster) to row-level clusters (28–1,214 observations per cluster)
substantially increases the statistical power available to H5 (see the
main text's power section). Resulting cluster sizes:

| Cluster | Unique ad groups | Row-level n |
|---|---|---|
| 0 | 39 | 1,214 |
| 1 | 41 | 217 |
| 2 | 13 | 28 |

**Attribution summary**: for each method, per-observation attributions
for the five engagement/spend features (CTR, CVR, Depth, log_cost,
log_impression — retained as "Group 0" purely as a reporting label) and
the two temporal features (hour_sin, hour_cos — "Group 1") are summarized
via a Gini coefficient, computed separately within each cluster.

**Statistical testing**:
- A **Kruskal-Wallis test** is computed once per method (not per feature),
  testing whether Group 0 attribution magnitude differs across the three
  clusters. All three methods reject the null (Individual SHAP: H=72.213,
  p<.0001; Permutation SHAP: H=143.831, p<.0001; Integrated Gradients:
  H=18.277, p=.0001) — this three-way agreement is the cross-method
  convergence the design is built to test directly.
- **Cross-method agreement** is quantified via pairwise Spearman ρ on
  cluster-level, feature-mean absolute attribution. The two Shapley-family
  methods agree strongly across all clusters (ρ = 0.857–0.964); agreement
  between the Shapley-family methods and the gradient-based Integrated
  Gradients is more variable (ρ = 0.607–0.893), which is expected and
  substantively informative — Shapley-based and gradient-based methods
  rest on different formal definitions of feature importance (cooperative-
  game marginal contribution vs. integrated local gradient), so partial
  rather than perfect convergence between method families is the expected
  pattern, not a failure of either method.

**H5 verdict**: supported. All three independently computed methods agree
that engagement/spend attribution differs significantly across ad-group
clusters, and all three agree that temporal (hour-of-day) attribution is
consistently more concentrated than engagement/spend attribution, across
every cluster.

## 8. Domain Shift and Adaptation (R2 — supplementary robustness check, not an independent hypothesis)

- **KS test** per feature, between Search- and Shopping-campaign
  observations. 6 of 7 features show p < .05 (all except the hour-cosine
  encoding, p = .068).
- **Frozen-encoder transfer**: a model trained on Search-campaign
  sequences has half its encoder parameters frozen, then is fine-tuned on
  Shopping-campaign sequences. The reported RMSE gain (naive transfer
  RMSE = 1.3176 → adapted transfer RMSE = 1.3128, +0.4%) is modest and is
  explicitly framed as providing empirical motivation for domain-adaptive
  design generally, not a claim that this specific recipe delivers a
  large performance gain in this instance.
- **Note on numbering**: this analysis is reported as **R2**, a
  robustness check corroborating H3's campaign-type moderation finding at
  the distributional level — it is not an independent sixth hypothesis
  ("H6") as an earlier draft numbered it.

## 9. External Validity (RQ6) — Leave-One-Advertiser-Out Cross-Validation

Generalization within the fixed-market, fixed-month, fixed-agency-provenance
scope of the study is assessed via **leave-one-advertiser-out (LOAO) cross-
validation across all 37 advertisers** in the panel — each fold withholds
one advertiser's sequences entirely and evaluates on that held-out
advertiser, using a GRU forecaster (hidden=128, layers=2, dropout=0.2).

- Mean RMSE = 1.2268, SD = 0.5789, n = 37 folds, min/max = 0.260/3.084.
- 95% CI of the mean fold RMSE (SE = 0.5789/√37 = 0.0952): **[1.040, 1.413]**.
- The mean RMSE is numerically close to, but should not be interpreted as
  replicating, the LSTM group-split test RMSE (1.2099) reported for H4b —
  the LOAO-CV forecaster (GRU) and the H4b winner (LSTM) are different
  architectures, so their proximity is at most suggestive that RMSE in the
  1.2–1.3 range is a stable property of this task, not evidence that the
  H4b result specifically survives resampling.
- Because each fold withholds an **entire advertiser** rather than a
  subset of hours or ad groups from a shared advertiser pool, this is a
  materially stronger generalization test than a within-advertiser,
  held-out-time-period validation would provide: it directly demonstrates
  generalization to advertisers the model has never seen during training,
  within the scope of this market, month, and agency-provenance.
- A supplementary regularization grid search (GRU, dropout × weight
  decay) identifies a best configuration (dropout=0.2, weight_decay=1e-4,
  RMSE=1.4073) as a robustness reference point; because this uses a
  different architecture/sweep than the headline LSTM result, it is not
  directly comparable to the H4b ranking.

> **Naming correction.** This cross-validation was previously described
> in some materials as "Leave-One-Ad-Group-Out" (LOGO-CV) across "37 ad
> groups" belonging to a single advertiser. The correct unit of
> cross-validation, consistent with the manuscript, is the **advertiser**:
> the panel contains 37 distinct advertisers, and each fold withholds one
> advertiser's data in full. Any reference elsewhere in the repository to
> "LOGO-CV" or "37 ad groups" in this context should be read as referring
> to this same leave-one-advertiser-out procedure.

## 10. Robustness Supplement (Weakness-Response Analyses)

A set of supplementary analyses were added during development in response
to anticipated reviewer concerns. Several of these predate the 37-advertiser
panel correction described above and are updated or reframed here
accordingly; all are exploratory and sit outside the core H1–H5/R1–R2/RQ6
hypothesis set unless otherwise noted.

1. **Augmentation-method quality**: KS/MMD stratification per augmentation
   method against real data confirms the three-method pool (β-VAE,
   Gaussian copula, moving-block bootstrap) balances fidelity against
   novelty rather than over-relying on any one generator (moving-block
   bootstrap has the highest fidelity but the least novelty; β-VAE the
   reverse). This supports, but is not a substitute for, the FSD quality
   gate reported in §4.
2. **Data-scaling curve**: RMSE as a function of the fraction of real
   training data used (20–100%, 3 seeds). Results at this scale do not
   show a clear monotonic improvement with more real data, suggesting the
   current sample has not yet reached a regime of strong returns to
   additional real data; read as scale-limited context for the
   augmentation design in §4, not as a claim about the augmentation
   pipeline's necessity.
3. **Cross-advertiser generalization**: superseded by the main-text RQ6
   leave-one-advertiser-out analysis (§9) — this is no longer a
   supplementary check but the paper's primary external-validity result,
   now that the panel provides 37 real (not synthetic) advertisers to
   generalize across. An earlier synthetic multi-advertiser simulation
   (perturbing CTR scale, cost scale, and ROAS noise to approximate
   between-advertiser heterogeneity) is retained in the codebase for
   reference but is no longer needed as a stand-in for real
   between-advertiser evidence, since RQ6 now provides that directly.
4. **Overfitting diagnostics**: train/validation gap and a dropout ×
   weight-decay grid search for the GRU architecture, corroborating the
   epoch-consistent domain-gap diagnostic reported as H4c in §5.
5. **Cold-start K-shot inference (ProtoNet)**: exploratory; see the scope
   note in §6. Not part of the core hypothesis set.
6. **DM multiple-comparison correction**: Bonferroni and BH-FDR correction
   across all 21 pairwise model comparisons is now integral to the H4b
   result itself (§5), not a separate supplementary check — the
   FDR-corrected 8-of-18 figure is the number reported as H4b's primary
   evidence, with the raw 13-of-18 figure reported alongside for
   transparency.
7. **IS/DSS theoretical grounding**: exploratory empirical tests of
   Signaling Theory and Resource Scarcity theory against this dataset are
   retained in the codebase but are not part of the core hypothesis set
   or the submitted manuscript's claims.

All supplementary results are reported honestly even when they complicate
the main narrative, consistent with the framework's overall commitment to
conservative, reviewer-anticipatory reporting.

## References

- Austin, P. C. (2011). An introduction to propensity score methods.
  *Multivariate Behavioral Research*, 46(3), 399–424.
- Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable
  distinction. *Journal of Personality and Social Psychology*, 51(6).
- Chamma, A., Thirion, B., & Engemann, D. (2024). Variable importance in
  high-dimensional settings requires grouping. *AAAI*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation.
  *ICML*.
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with
  selective state spaces. *arXiv:2312.00752*.
- Hoenig, J. M., & Heisey, D. M. (2001). The abuse of power: The
  pervasive fallacy of power calculations for data analysis. *The
  American Statistician*, 55(1), 19–24.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. *NeurIPS*.
- Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and
  individual predictions with feature contributions. *Knowledge and
  Information Systems*, 41(3), 647–665.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for
  deep networks. *ICML*.
- Zhao, X., Lynch, J. G., Jr., & Chen, Q. (2010). Reconsidering Baron and
  Kenny: Myths and truths about mediation analysis. *Journal of Consumer
  Research*, 37(2), 197–206.
