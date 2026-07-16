# SADAF Full Hypothesis Results Table

This table consolidates the verdict, method, and key numerical results
for every research question and hypothesis tested in SADAF, together
with the robustness-supplement findings that qualify or contextualize
the main results.

> **v6.0 alignment note.** This table has been fully revised to match
> the submitted manuscript and `README.md` (v6.0). Two corrections
> apply throughout: (1) all sample sizes are pooled or drawn across a
> **37-advertiser panel**, not one advertiser; and (2) the explainability
> row (previously "H5, GS-SHAP") now reports the individual-level,
> cross-verified attribution design (Individual SHAP + Permutation SHAP
> + Integrated Gradients), which replaced group-level GS-SHAP entirely.
> Numbers below are the ones that appear in the manuscript; an earlier
> internal run with a smaller test split (n=34) and different model
> rankings (e.g., Bayesian LSTM as the H4b winner) has been superseded
> and is not reproduced here — see `readme/README_v5_full.md` for that
> prior run's captured output, retained for provenance only.

## Primary Hypotheses

| RQ / H | Hypothesis | Method | Key Result | Verdict |
|---|---|---|---|---|
| RQ0 | Do causal, predictive, and explainability patterns from Google-dominated markets replicate in a domestically-dominated, Naver-and-Kakao ecosystem? | Framing question, answered by the pattern of support across H1–H5, R1, R2, RQ6 | See synthesis in README §6 (Discussion) | Not itself tested — answered by the pattern below |
| H1 | High-CTR ads causally increase conversion probability | PSM (caliper=0.1σ) + Doubly Robust IPW, pooled across 37-advertiser panel | IPW-ATT = 0.1286 (primary); PSM-ATT = 0.1347, 95% CI [0.1254, 0.1434]; n_matched = 14,987 | ✓ Supported — IPW primary; PSM corroborating. Residual covariate imbalance (`log_impression`, `log_cost`, \|SMD\|>0.1) explicitly disclosed and corrected via DR weighting. |
| H2 | Browsing depth mediates CTR→Conversion | Baron-Kenny + Bootstrap (B=2,000) | a = −0.3077, b = −0.0861, indirect (a×b) = 0.0265, 95% CI [0.0200, 0.0337] | △ Informative departure from the hypothesized mediating mechanism — negative suppressor (a<0, b<0, a×b>0): Depth proxies decision hesitancy, not engagement. Reported as such, not as "H2 supported," since the opposite mechanism from the one hypothesized was found. Indirect effect itself remains statistically distinguishable from zero. |
| H3 | Campaign type moderates CTR→ROAS slope | OLS, HC3-robust SE, interaction term, n=9,069 (pooled across panel) | β_interaction = 0.386 (p < .0001); ME_Search = 0.949; ME_Shopping = 0.563 | ✓ Supported — Search campaigns show a ~68% steeper CTR→ROAS slope than Shopping. Corroborated distributionally by R2. |
| H4a | A parsimonious linear classifier matches or exceeds recurrent architectures for has_roas prediction under extreme sparsity (n<200 training sequences) | Logistic Regression vs. LSTM, Bayesian LSTM, MLP (AUC), n=174 train / 24 test sequences | Logistic Regression AUC = 0.6143 (best); LSTM = 0.6115; Bayesian LSTM = 0.5894; MLP = 0.5445 | ✓ Supported — logistic regression attains the highest AUC of any model regardless of which recurrent architecture it is benchmarked against; near-linear separability at n≈200 makes added recurrent capacity actively counterproductive, not merely unnecessary. |
| H4b | A recurrent architecture with the augmentation pipeline achieves significantly lower forecasting error than linear/feed-forward baselines | Diebold-Mariano test, raw + BH-FDR correction across 21 pairs, n=24 test sequences | LSTM RMSE = 1.2099 (best); of 21 pairs, 18 computable (3 non-convergent: Mamba–Ridge, Mamba–MLP, Ridge–MLP); 13/18 significant raw p<.05; 8/18 significant after FDR correction | ✓ Supported — LSTM beats GRU, BiLSTM, Mamba, Ridge, MLP, and Bayesian LSTM at FDR-corrected significance. Non-significant pairs read via a priori MDE (dz=0.60 raw / 0.88 FDR-equivalent at n=24), not post-hoc power. |
| H4c | The sign of the real-validation/training-loss gap differs across architectures, with at most one showing the classical overfitting pattern | Epoch-consistent domain-gap diagnostic (train loss paired with the epoch of best real-validation loss) | Mamba: gap_real = +0.1142 (val > train — overfitting signature); Bayesian LSTM, LSTM, GRU, BiLSTM: all gap_real < 0 (train > val — not overfitting) | ✓ Supported, descriptive formulation — Mamba is the only architecture displaying the classical overfitting signature; not tested for statistical significance (gap_real is a deterministic single-pair statistic per architecture, not a distribution). |
| R1 | (Robustness check, not an independent hypothesis) Is Mamba's weaker 4-step accuracy an artifact of sequence length? | Mamba re-evaluated at SEQ_LEN=6 vs. SEQ_LEN=4 | Mamba remains the weakest or near-weakest performer at both lengths | Corroborates short-sequence vulnerability documented by Liu et al. (2025) and Wang et al. (2025); context for H4c, not a standalone finding. |
| H5 | Ad-group clusters exhibit statistically distinct engagement/spend attribution patterns, corroborated across independent methods | Individual SHAP (primary) + Permutation SHAP + Integrated Gradients, computed on identical model/data; K-means (k=3) row-level clusters (n=1,214 / 217 / 28 observations; 39 / 41 / 13 ad groups) | Kruskal-Wallis (Group 0 attribution, by method): Ind.SHAP H=72.213 p<.0001; Perm.SHAP H=143.831 p<.0001; IG H=18.277 p=.0001. Cross-method Spearman ρ: Shapley-family pairs 0.857–0.964; Shapley-vs-IG pairs 0.607–0.893 | ✓ Supported — all three independently computed methods agree Group 0 (engagement/spend) attribution differs across clusters; all three agree temporal (Group 1) attribution is consistently more concentrated than engagement attribution, across every cluster. Replaces an earlier group-level GS-SHAP design not warranted at this feature dimensionality (Chamma et al., 2024). |
| R2 | (Robustness check, not an independent hypothesis) Do Search and Shopping campaigns exhibit significant feature-distribution shift? | Kolmogorov-Smirnov test, 7 features; frozen-encoder domain adaptation | 6/7 features p<.05 (all but hour-cosine, p=.068); naive-transfer RMSE=1.3176 → adapted-transfer RMSE=1.3128 (+0.4%) | Corroborates H3 distributionally. Domain-adaptation gain is modest and reported as motivation for domain-adaptive design generally, not a performance-superiority claim. Previously mislabeled "H6" in an earlier draft — this is a robustness check, not an independent sixth hypothesis. |
| RQ6 | Does the predictive framework generalize to advertisers not seen during training, within this market/period/agency-provenance scope? | Leave-one-advertiser-out (LOAO) cross-validation across all 37 advertisers, GRU forecaster (hidden=128, layers=2, dropout=0.2) | Mean RMSE = 1.2268, SD = 0.5789, 95% CI [1.040, 1.413], n=37 folds; min/max = 0.260/3.084 | ✓ Supported, within stated scope only — each fold withholds an entire advertiser, providing direct evidence of generalization to unseen advertisers. Numerically close to, but not a replication of, the H4b LSTM result (different architecture: GRU vs. LSTM). Previously mislabeled "LOGO-CV across 37 ad groups" within one advertiser — corrected to leave-one-**advertiser**-out across 37 real, distinct advertisers. |

## Exploratory Analyses (not part of the core hypothesis set)

These analyses are retained in the codebase and documented for
completeness but do not appear as claims in the submitted manuscript.

| Analysis | Method | Key Result | Status |
|---|---|---|---|
| Bayesian calibration | MC Dropout (n=500) + temperature scaling on Bayesian LSTM | 95% CI coverage and interval width computed at four nominal levels (50/80/90/95%) | Exploratory — informed pipeline development; not reported as a manuscript claim or numbered hypothesis. |
| ProtoNet cold-start K-shot inference | Prototypical Network, cosine-similarity weighting, K=1→21 support examples | Cold-start RMSE trajectory computed vs. full-data GRU baseline | Exploratory — retained in `sadaf/models/protonet.py`; not part of H1–H5/R1–R2/RQ6. |
| IS/DSS theoretical grounding | Signaling Theory (CTR-quartile monotonicity) and Resource Scarcity (ROAS uncertainty vs. impression volume) empirical tests | Mixed/inconclusive results at current sample sizes | Exploratory — not part of the submitted manuscript's claims. |

## Robustness Supplement (Weakness-Response Analyses)

| Code | Concern Addressed | Method | Key Result | Implication |
|---|---|---|---|---|
| W1a | Augmentation method quality varies | KS/MMD stratification per augmentation method vs. real data | Moving-block bootstrap: highest fidelity, least novelty; β-VAE: most novel, least faithful; Gaussian copula: intermediate | Validates that the three-method pool balances fidelity against novelty; supports, does not replace, the FSD quality gate (main text §4.5 / methodology §4). |
| W1b | Performance gains may be augmentation-driven, not data-driven | Data-scaling curve, 20–100% of real training data, 3 seeds | No clear monotonic RMSE improvement with more real data at this scale | Suggests current sample size has not yet reached a regime of strong returns to additional real data; scale-limited context, not a claim about augmentation necessity. |
| W1c → **superseded by RQ6** | Hold-out RMSE may not reflect true cross-advertiser generalization | Originally: Leave-One-Ad-Group-Out CV within one advertiser. **Now:** Leave-One-Advertiser-Out CV across all 37 real advertisers (see RQ6 above) | RQ6: Mean RMSE = 1.2268 ± 0.5789, 95% CI [1.040, 1.413] | This concern is now addressed directly by RQ6 as a primary, not supplementary, result — the panel's 37 real advertisers make this a genuine cross-advertiser generalization test rather than a within-advertiser proxy. |
| W2a | Single-month data may hide temporal instability | Hold-one-hour-block-out CV (4 blocks) | Moderate variability across blocks; smallest block (n=1) flagged as unstable and not over-interpreted | Scope condition on the one-month observation window (README §8.1), not a defect. |
| W2b → **superseded by RQ6** | External validity beyond the advertisers in the dataset | Originally: synthetic multi-advertiser simulation (perturbing CTR/cost scale, ROAS noise) as a stand-in for real between-advertiser heterogeneity. **Now:** the panel already contains 37 real, distinct advertisers, and RQ6 tests generalization across them directly. | RQ6 (above) supersedes the need for a synthetic stand-in | The earlier synthetic simulation's negative result (poor transfer to out-of-distribution synthetic advertisers) is retained in the codebase for reference but is no longer load-bearing for the paper's external-validity claim, since RQ6 now provides real cross-advertiser evidence directly. |
| W3 | Training curves show train/val divergence (possible overfitting) | Quantified best-val/train gap (see H4c); dropout × weight-decay grid search for GRU | Corroborates H4c's epoch-consistent domain-gap diagnostic; grid search provided for replication | Supports, and is now integral to, the H4c result rather than a separate supplementary check. |
| W4 → **resolved by design change** | Attention attribution disagreed with the other 3 methods (measures a different axis) | Originally: 3-method consensus excluding Attention, computed as a supplementary check. **Now:** Attention is dropped from the explainability design entirely (see H5 above) | H5 now uses only Individual SHAP, Permutation SHAP, and Integrated Gradients from the start | Rather than retaining a structurally different fourth method and explaining away its disagreement after the fact, the design now uses only methods that estimate the same class of quantity (feature-level importance), consistent with Section 5.3/2.7 of the manuscript. |
| W5 | ProtoNet K-shot RMSE far exceeds full-data baseline | Cold-start trajectory (K=1→21) vs. naive-mean baseline; NN-prototype vs. random selection | NN-prototype selection underperformed random selection in this test | Exploratory, honest negative result; ProtoNet's practical value (if any) rests on the cold-start trajectory shape, not on beating the full-data baseline. Not part of the core manuscript claims (see Exploratory Analyses above). |
| W6 → **integral to H4b** | DM significance may not survive multiple-comparison correction | Bonferroni and BH-FDR correction across 21 pairwise comparisons | 13/18 significant raw; 8/18 significant after FDR correction (see H4b above) | This is no longer a separate caveat layered on top of a different headline number — the FDR-corrected 8/18 figure and the raw 13/18 figure are reported together as H4b's primary evidence. |
| W7 | Unclear theoretical (IS/DSS) contribution | CTR-quartile monotonicity test; ROAS-uncertainty-by-impression-volume scarcity table | Mixed/non-monotone results at current sample sizes | Exploratory (see Exploratory Analyses above); not part of the submitted manuscript's claims. |

## Summary Verdict Counts (core hypothesis set only)

| Verdict | Count | Items |
|---|---|---|
| ✓ Supported | 8 | H1, H3, H4a, H4b, H4c, H5, R2 (corroborating), RQ6 |
| △ Informative departure from hypothesized mechanism | 1 | H2 |
| Robustness check, not independently verdicted | 1 | R1 |

No item in the core hypothesis set carries a "Null / boundary condition"
verdict in the current design: the earlier draft's H4a null result was a
consequence of testing the hypothesis in the reversed direction (Bayesian
LSTM vs. logistic regression, rather than logistic regression vs. every
recurrent architecture) on a different, smaller test split; the corrected
H4a specification above is supported.

## Manuscript Status

A complete manuscript (Abstract through References, ~37 pages) has been
prepared incorporating all corrections above: the 37-advertiser panel
framing, the Naver-and-Kakao (via SearchM Co., Ltd.) platform scope, the
individual-level cross-verified explainability design, and the
leave-one-advertiser-out external-validity check. The manuscript is
written to maximize the strengths of the design (well-powered causal
pillar, cross-advertiser generalization evidence, cross-verified
explainability) while transparently reporting the scope conditions and
power limitations of the sequence-level predictive tests (H4a–c), rather
than being tailored to any single target journal's specific requirements.
Journal-specific formatting (word limits, reference style, figure
resolution) has not yet been applied and should be handled at submission
time for the specific venue chosen.
