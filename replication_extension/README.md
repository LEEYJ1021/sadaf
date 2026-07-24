# Replication Extension: Five-Seed Stability, Real-Only Ablation, and Shopping-Only Baseline

This folder collects the three follow-up analyses that revise SADAF's regression-stage
(H4b) and domain-adaptation (R2) results. They were run in response to reviewer
comments asking (1) whether the single-split "LSTM is best" result holds up under
different random seeds, (2) whether the augmentation pipeline is actually improving
forecast accuracy or doing something else, and (3) whether the domain-adaptation gain
reported for Search→Shopping transfer reflects genuine domain shift or is simply how
hard Shopping-campaign forecasting is on its own. None of these three analyses were in
the original SADAF pipeline; they extend it without modifying any existing script.

All three reuse the identical group-aware train/validation/test split (or, for the
Shopping-only check, the identical Shopping-campaign split) used elsewhere in SADAF, so
results here are directly comparable to the main README's Table 7 and Section 6.8
numbers.

## What's here

```
replication_extension/
├── README.md                                  ← this file
├── scripts/
│   ├── 10_multiseed_stability.py               five-seed replication of Table 7
│   ├── 11_augmentation_ablation.py              real-only vs. augmented ablation
│   ├── 12_shopping_only_baseline.py             Shopping-only forecaster baseline
│   ├── fig3_framework_architecture.py           regenerates the updated architecture figure
│   ├── fig4_regression_fiveseed.py              regenerates the five-seed RMSE figure
│   └── fig5_dm_heatmap.py                       regenerates the full 7×7 DM heatmap
├── logs/
│   ├── 10_multiseed_stability.log               captured stdout, 5-seed run
│   ├── 11_augmentation_ablation.log             captured stdout, real-vs-augmented run
│   └── 12_shopping_only_baseline.log            captured stdout, 3-seed Shopping run
└── figures/
    ├── fig4_regression_fiveseed.png             Figure — five-seed RMSE distribution vs. Seed-42
    ├── fig5_dm_heatmap.png                      Figure — full 21-pair DM heatmap (raw + BH-FDR)
    ├── table7_multiseed_raw.csv                 per-seed, per-architecture RMSE (n=5 seeds × 7 models)
    ├── table7_multiseed_raw_summary.csv         cross-seed mean/SD/rank-1% per architecture
    ├── table_ablation_raw.csv                   per-seed RMSE, real-only vs. augmented (7 models)
    ├── table_ablation_raw_summary.csv           mean real vs. augmented RMSE and % change
    ├── table_ablation_raw_wilcoxon.csv          paired Wilcoxon signed-rank tests (n=5 seeds)
    ├── table_ablation_raw_dm.csv                per-seed Diebold–Mariano (augmented vs. real)
    ├── table_shopping_only_raw.csv              per-seed naive / adapted / Shopping-only RMSE
    ├── table_shopping_only_raw_summary.csv      decomposition of the domain-shift gap
    └── table_shopping_only_raw_dm.csv           pairwise DM tests across the three conditions
```

## 1. Five-seed stability check (`10_multiseed_stability.py`)

**Question:** Does the Table 7 ranking (LSTM lowest RMSE at Seed 42) hold across other
random seeds?

**Design:** The group-aware train (n=174) / validation (n=24) / test (n=24) split is
built once and reused for every seed; only model initialization, dropout, minibatch
shuffling, and the stochastic augmentation draw vary. Five seeds (42, 1, 7, 123, 2024)
are run end-to-end, each retraining the reference GRU used for augmentation and
regenerating the ~870-sequence augmented training corpus from scratch.

**Result:** It does not. LSTM's Seed-42 RMSE (1.2099) is the best score of any
model/seed combination in the whole panel, but LSTM ranks #1 in only 1 of 5 seeds
(20%) and has the second-largest cross-seed SD (0.1631). **Bayesian LSTM** has the
lowest five-seed mean RMSE (1.3604), the smallest SD (0.0240), and ranks #1 in 2 of 5
seeds (40%) — a materially more stable result. Kendall's W across the 5-seed × 7-model
ranking is 0.6514 (moderate, not near-1 concordance), and a Friedman test confirms
architecture RMSEs are not interchangeable across the seed panel (χ²=19.54, p=.0033).
See `figures/fig4_regression_fiveseed.png`.

**Reading:** the *architecture-specific* claim "LSTM is the single best forecaster" is
a Seed-42 artifact, not a stable finding. The broader claim "recurrent/gated
architectures beat linear and feed-forward baselines" survives — Bayesian LSTM, GRU,
BiLSTM, and LSTM all outperform Ridge and MLP in five-seed mean RMSE.

## 2. Real-only vs. augmented ablation (`11_augmentation_ablation.py`)

**Question:** Is the β-VAE + Gaussian-copula + moving-block-bootstrap augmentation
pipeline actually improving forecast accuracy, or doing something else?

**Design:** For each of the same five seeds, every architecture is trained twice on the
identical split: once on the 174 real training sequences only ("real-only"), and once
on the ~870-sequence augmented corpus ("augmented"). Both conditions are evaluated on
the same held-out real test set, so the two RMSEs are directly comparable per seed and
per architecture.

**Result:** For six of seven architectures, augmentation slightly *increases* mean
RMSE (BayesianLSTM +0.12, LSTM +0.11, GRU +0.04, BiLSTM +0.05, Mamba +0.19, Ridge
+0.23), though none of these six differences is significant at n=5 seeds (Wilcoxon
p≥.0625, the minimum attainable two-sided p at this sample size). The sole exception is
MLP, where augmentation cuts mean RMSE by 2.60 (real-only mean RMSE 4.43 vs. augmented
1.83) — a Wilcoxon p=.0625, the smallest attainable value at n=5. Inspecting the
real-only MLP RMSEs directly (4.17–4.74 across the five seeds) shows this is
catastrophic overfitting on 174 real sequences being *prevented* by augmentation, not
an already-adequate model being further improved.

**Reading:** in this dataset, the augmentation pipeline's primary demonstrated role is
regularization for overfitting-prone architectures (MLP specifically), not a general
accuracy improvement for recurrent forecasters that are already well suited to short
sequences. This directly qualifies the FSD "distributional fidelity" diagnostic
reported elsewhere in SADAF: passing the Fréchet Sequence Distance gate confirms the
synthetic sequences resemble the real ones distributionally, but distributional
fidelity and downstream predictive improvement are shown here to be distinct
properties.

## 3. Shopping-only baseline (`12_shopping_only_baseline.py`)

**Question:** SADAF's domain-adaptation check (Section 6.8 of the main README) reports
a naive Search→Shopping transfer RMSE and a fine-tuned "adapted" RMSE, but has no
condition trained (and only ever trained) on Shopping data. Without that, the
naive-vs-adapted gap can't be decomposed into "cost of genuine domain shift" vs.
"Shopping forecasting is just hard on its own."

**Design:** Three conditions are evaluated on the identical, once-built Shopping-test
split, across 3 seeds (42, 1, 7):
1. **Naive** — GRU trained only on Search, evaluated directly on Shopping-test.
2. **Adapted** — the same Search-trained GRU, with the later half of its encoder + the
   output head fine-tuned on Shopping-train (50% frozen), matching the main pipeline's
   recipe exactly.
3. **Shopping-only** — a fresh GRU trained from scratch on Shopping-train alone, never
   exposed to Search data, evaluated on the same Shopping-test set.

**Result (mean across 3 seeds):** naive RMSE = 1.3163, adapted RMSE = 1.3114,
Shopping-only RMSE = 1.3115. The **domain-shift recoverable gap** (naive − Shopping-only)
is only 0.0048, and the **adaptation-captured gap** (naive − adapted) is 0.0050 — an
adaptation-capture ratio of ~1.06, i.e., the fine-tuning recipe captures essentially
all of the (very small) recoverable domain-shift cost. Naive vs. adapted is
significant after FDR correction at all 3 seeds (p_bh ≤ .0033); naive vs.
Shopping-only is significant at 2 of 3 seeds; adapted vs. Shopping-only is not
significant at any seed.

**Reading:** almost all of the ~1.31 RMSE observed under naive transfer reflects the
intrinsic difficulty of Shopping-campaign log-ROAS forecasting itself, not a
recoverable cost of having been trained on the wrong campaign type. The domain
adaptation recipe is not being shortchanged by a stronger untried alternative — it is
already close to the Shopping-only floor. This narrows, rather than undermines, the
domain-adaptation motivation reported in the main README: the KS-test evidence for
distributional shift between Search and Shopping features (Section 6.8) is real, but
that shift turns out to carry very little exploitable forecasting cost for this
particular architecture and task.

## How these results feed back into the main README

- Table 7 (regression-stage results) in the main README now reports both the
  preregistered Seed-42 single split and the five-seed replication side by side, and
  revises the headline claim from "LSTM is the best forecaster" to "gated recurrent
  architectures beat linear/feed-forward baselines; Bayesian LSTM is the most stable
  single architecture."
- Section 6.5/6.6 (augmentation and overfitting) now report the real-only ablation
  result and reframe the augmentation pipeline's contribution as primarily
  regularization rather than general accuracy improvement.
- Section 6.8 (domain adaptation) now discloses the Shopping-only decomposition and
  notes that the adapted-vs-naive gain, while statistically real, recovers nearly all
  of a small domain-shift cost rather than leaving headroom for a better recipe.
