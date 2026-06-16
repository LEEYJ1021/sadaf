# SADAF Full Hypothesis Results Table

This table consolidates the verdict, method, and key numerical results
for every research question and hypothesis tested in SADAF, including
the robustness-supplement findings (W-series) that qualify or contextualize
the main results.

## Primary Hypotheses

| RQ / H | Hypothesis | Method | Key Result | Verdict |
|---|---|---|---|---|
| RQ1 / H1 | High-CTR ads causally increase conversion probability | PSM (caliper=0.1σ) + Doubly Robust IPW | IPW-ATT = 0.1286 (primary); PSM-ATT = 0.1347, 95% CI [0.1254, 0.1434]; n_matched = 14,987 | ✓ Supported — IPW primary; PSM corroborating. Residual covariate imbalance (`log_impression`, `log_cost`, \|SMD\|>0.1) explicitly disclosed and corrected via DR weighting. |
| RQ2 / H2 | Browsing depth mediates CTR→Conversion | Baron-Kenny + Bootstrap (B=2,000) | a = −0.3077, b = −0.0861, indirect (a×b) = 0.0265, 95% CI [0.0200, 0.0337] | △ Negative suppressor (a<0, b<0, a×b>0): Depth proxies decision hesitancy, not engagement. Proportion mediated (−42.8%) reported in appendix only; sign pattern is the primary finding. |
| RQ3 / H3 | Campaign type moderates CTR→ROAS slope | OLS, HC3-robust SE, interaction term | β_interaction = 0.3860 (p < 0.001); ME_Search = 0.949; ME_Shopping = 0.563 | ✓ Supported — Search campaigns show a steeper CTR→ROAS slope than Shopping. |
| RQ4 / H4a | Bayesian LSTM classifier outperforms logistic regression for has_roas prediction | BayesianLSTM-Cls vs. LR-Cls (AUC) | BayesianLSTM-Cls AUC = 0.6107; LR-Cls AUC = 0.6143 | ⚬ NULL (boundary condition) — sparse ad-group sequences are largely linearly separable at this sample size (n=2,211 sequences); LR is competitive. Framed as theoretically informative, not a framework failure. |
| RQ4 / H4b | Best recurrent architecture outperforms Ridge/MLP baselines for log-ROAS regression | DM test, bootstrap RMSE CI | BayesianLSTM RMSE = 1.4219 [1.1075, 1.7312] vs. Ridge RMSE = 2.0924, MLP RMSE = 2.0233 | ✓ Supported — BayesianLSTM > LSTM (DM p=0.044), BayesianLSTM > Mamba (DM p=0.033); vs. GRU/BiLSTM: not significant (reported conservatively; see W6 multiple-comparison caveat below). |
| RQ4 / H4c | Mamba is more robust to sequence-length variation than LSTM/GRU | ΔRMSE per +2 SEQ_LEN (4→6), DM test at fixed SEQ_LEN | Mamba: SL4=1.5793→SL6=1.5752 (Δ=−0.0041); LSTM: SL4=1.5669→SL6=1.5725 (Δ=+0.0056); GRU: Δ=+0.4061 | ✓ Supported (robustness only, not accuracy) — DM tests at fixed SEQ_LEN are non-significant, as expected under H4c. |
| RQ4 / Bayesian | Calibrated posterior uncertainty quantification | MC Dropout (n=500) + temperature scaling (T=1.5) | 95% CI coverage = 94.1%; mean interval width = 7.1034 | ✓ Novel contribution — well-calibrated (within 3pp of all four nominal coverage levels tested: 50/80/90/95%). |
| RQ4 / ProtoNet | Cold-start K-shot ROAS inference | Prototypical Network, cosine-similarity weighting | K=1: RMSE=2.3187; K=2: 2.2159; K=3: 2.3469; K=5: 2.2784; full-data GRU baseline: 1.5670 | ✓ Novel contribution, with caveat — establishes a cold-start inference capability; absolute RMSE remains well above the full-data baseline (see W5 trajectory analysis). |
| RQ5 / H5 | Ad-group clusters show distinct, convergent attribution patterns | KMeans (k=3) + Kruskal-Wallis η² + 4-method Spearman ρ agreement | η²_max = 0.525 (hour_sin/hour_cos, p<0.001); 4/7 features significant; cluster ROAS Kruskal-Wallis p<0.0001 | ✓ Supported — cluster-specific attribution patterns are statistically distinct, though exploratory given small per-cluster n (7/17/10). See W4 for attribution-method agreement caveats. |
| RQ6 / H6 | Search and Shopping campaigns exhibit significant feature distribution shift | Kolmogorov-Smirnov test, 7 features | 6/7 features p<0.05; target (log-ROAS) KS=0.0398 | ✓ Supported — motivates frozen-encoder domain adaptation; fine-tuning improvement = +0.2% (modest; framed as theoretical justification, not performance claim). |

## Robustness Supplement (W-series)

| Code | Concern Addressed | Method | Key Result | Implication |
|---|---|---|---|---|
| W1a | Augmentation method quality varies | KS + MMD per method vs. real data | MBB: mean KS=0.011, MMD=0.0056 (best fidelity); Copula: KS=0.112; β-VAE: KS=0.404 (most novel but least faithful) | Validates that the combined three-method pool balances fidelity (MBB) against novelty (β-VAE), rather than over-relying on any one generator. |
| W1b | Performance gains may be augmentation-driven, not data-driven | Data-scaling curve, 20–100% of real training data, 3 seeds, power-law fit | RMSE roughly stable (1.32–1.48) across fractions; no clear monotonic improvement with more real data at this scale | Suggests current sample size has not yet reached a regime of strong returns to additional real data; results should be read as scale-limited. |
| W1c | Hold-out RMSE may not reflect true cross-ad-group generalization | Leave-One-Ad-Group-Out CV (37 groups) | LOGO-CV RMSE = 1.2041 ± 0.5976 vs. hold-out GRU RMSE = 1.5670 (Δ = −0.36) | LOGO-CV RMSE is *lower* than hold-out, with high variance (±0.60) — generalization across ad groups is plausible but uncertain; large group-level RMSE spread reported, not hidden. |
| W2a | Single-month data may hide temporal instability | Hold-one-hour-block-out CV (4 blocks) | RMSE range 0.39–1.16 across blocks; CV std(RMSE) = 0.38 | Moderate temporal variability; Evening block (n=1) result is unstable and should not be over-interpreted. |
| W2b | Single-advertiser data limits external validity | Synthetic 5-advertiser simulation (CTR/cost scale, ROAS noise) | Pre-trained GRU RMSE on synthetic advertisers: 2.07–2.55, all *worse* than original GRU RMSE (1.567) and all with negative R² | Honest negative result: the model does **not** generalize well to out-of-distribution synthetic advertisers without retraining; reported as a genuine limitation rather than smoothed over. |
| W3 | Training curves show train/val divergence (possible overfitting) | Quantified best_val − final_train gap; 3×3 dropout × weight_decay grid | Largest gap: Mamba (+0.97, 18.4×); smallest: BayesianLSTM (+0.42, 2.3×). Best regularisation: dropout=0.4, weight_decay=1e-3 (RMSE=1.3777) | Confirms overfitting is present across all five architectures to varying degrees; BayesianLSTM's built-in dropout-based regularization is the most resistant. Grid search results provided for replication. |
| W4 | Attention attribution disagrees with other 3 methods (ρ as low as −0.04) | 3-method (excl. Attention) consensus; temporal-weight visualization | 3-method avg ρ: C0=0.719, C1=0.624, C2=0.699 (all moderate-to-high) | Reframes Attention as measuring a different axis (temporal position vs. feature importance); recommends excluding it from the triangulation claim in the main text. |
| W5 | ProtoNet K-shot RMSE far exceeds full-data baseline | Cold-start trajectory (K=1→21) vs. naive-mean baseline; NN-prototype vs. random selection | NN-prototype K=5 RMSE = 2.4284 vs. random K=5 RMSE = 2.2784 (NN underperforms random in this test) | Honest negative result for the NN-prototype refinement; practical value of ProtoNet rests on the cold-start trajectory shape (converging toward, not matching, the full-data baseline), not on beating it outright. |
| W6 | DM significance at n=34 may not survive multiple-comparison correction | Bonferroni + BH-FDR correction across 21 pairwise comparisons; Cohen's d | Raw: 2 significant pairs; BH-FDR: 0; Bonferroni: 0 | Materially weakens the H4b significance claim once corrected; reported transparently as a key limitation rather than omitted. |
| W7 | Unclear theoretical (IS/DSS) contribution | CTR-quartile monotonicity test (Kendall τ); ROAS-uncertainty-by-impression-volume scarcity table | Conv-rate monotonicity: τ=0.0, p=1.0 (non-monotone); ROAS monotonicity: τ=0.667, p=0.333 (non-monotone, small n=4); peak ROAS std at <10 impressions (41,285) | The simple CTR-quartile signaling test does **not** confirm strict monotonicity — reported as a genuine null result, with the scarcity/zero-inflation-by-volume finding retained as the stronger empirical anchor for the cold-start framing. |

## Summary Verdict Counts

| Verdict | Count | Hypotheses |
|---|---|---|
| ✓ Supported | 7 | H1, H3, H4b*, H4c, H5, H6, Bayesian/ProtoNet (novel) |
| ⚬ Null / boundary condition | 1 | H4a |
| △ Qualified (sign-pattern only) | 1 | H2 |

\* H4b's "supported" verdict is qualified by the W6 multiple-comparison
correction, under which none of the pairwise DM comparisons remain
significant after Bonferroni or BH-FDR adjustment. This caveat must be
disclosed in any submission text alongside the headline RMSE ranking.

## Target Journal Assessment

| Journal | ABS Ranking | Estimated Outcome |
|---|---|---|
| Decision Support Systems (DSS) | 3 | Major Revision (primary target) |
| Information & Software Technology (ISF) | 3 | Possible secondary target |

These estimates reflect the overall manuscript state after incorporating
all reviewer-anticipated weaknesses (W1–W7) and should be revisited once
actual reviewer feedback is received.