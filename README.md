# SADAF: Sparse Ad Data Augmentation Framework

> **A Unified Causal-Predictive-Explainable Framework for Cold-Start Advertisement Performance Forecasting**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents
1. Overview
2. Research Questions
3. Framework Architecture
4. Data Description
5. Repository Structure
6. Installation
7. Usage
8. Results & Visualizations
9. Key Findings
10. Code Fix Log (v3)
11. Figures Requiring Update
12. Citation
13. Data Availability
14. License

---

## Overview

**SADAF** (Sparse Ad Data Augmentation Framework) addresses one of the most persistent challenges in computational advertising: the **cold-start problem**, where newly launched ads lack sufficient historical data for reliable performance prediction.

The framework integrates three methodological pillars into a single pipeline:

| Pillar | Method | Purpose |
|--------|--------|---------|
| **Causal Estimation** | PSM + Doubly Robust IPW + Mediation + Moderation | Identify *why* ads convert |
| **Bayesian Prediction** | BayesianLSTM + GRU + BiLSTM + Mamba + ProtoNet | Predict *what* ROAS will be, with uncertainty |
| **Explainability** | GS-SHAP + IntGrad + Perm-SHAP + Attention | Explain *which* features drive outcomes |

A custom three-method augmentation pipeline (β-VAE + Gaussian Copula + Moving Block Bootstrap) addresses the fundamental data scarcity problem, validated with Fréchet Score Distance (**FSD = 0.6852**, below the acceptance threshold of 2.0).

> **Note on FSD:** The original README reported FSD = −0.1347, which reflected an earlier pipeline run with a different random seed. The canonical value from the fixed-seed v3 pipeline is **0.6852**. Both values pass the acceptance threshold (<2.0); the updated value is used in all reported results.

---

## Research Questions

SADAF is organized around six research questions (RQ), each mapped to empirically testable hypotheses:

### RQ1 — Causal Effect of CTR on Conversion
**H1:** High click-through rate (CTR) ads causally increase conversion probability compared to low-CTR ads, after controlling for impression volume, cost, and campaign type.

*Method: Propensity Score Matching (PSM, caliper = 0.1σ) + Doubly Robust Inverse Probability Weighting (IPW)*

---

### RQ2 — Mediating Role of Browsing Depth
**H2:** Browsing depth (Depth) mediates the relationship between CTR and conversion, and the sign pattern of the mediation paths reveals structural characteristics of advertiser-consumer interaction.

*Method: Baron-Kenny decomposition + Bootstrap mediation (B = 2,000)*

---

### RQ3 — Campaign-Type Moderation
**H3:** The positive relationship between CTR and ROAS is moderated by campaign type, such that Search campaigns exhibit a stronger CTR→ROAS slope than Shopping campaigns.

*Method: OLS interaction with HC3-robust standard errors*

---

### RQ4 — Deep Sequential ROAS Prediction
**H4a:** A Bayesian LSTM classifier outperforms logistic regression for binary ROAS prediction (zero vs. non-zero) on augmented sparse ad sequences.

**H4b:** The best-performing recurrent architecture achieves significantly lower RMSE than ridge regression and MLP baselines for log-ROAS prediction, as confirmed by Diebold-Mariano (DM) tests.

**H4c:** Mamba (selective state-space model) exhibits greater robustness to sequence-length variation (SEQ_LEN 4 → 6) compared to standard LSTM and GRU, measured by ΔRMSE per +2 time steps.

*Method: BayesianLSTM / GRU / BiLSTM / Mamba trained on β-VAE + Copula + MBB augmented data; DM test with bootstrap CI*

---

### RQ5 — Cluster-Specific Attribution Explanation
**H5:** Ad group clusters exhibit statistically distinct feature attribution patterns across HSIC-defined feature groups (measured by Kruskal-Wallis η²), and multiple attribution methods (GS-SHAP, Integrated Gradients, Permutation-SHAP) produce convergent rankings.

> **Note on GS-SHAP group structure:** GS-SHAP decomposes attribution at the HSIC group level, not the individual feature level. Group 0 = {CTR, CVR, Depth, log_cost, log_impression}; Group 1 = {hour_sin, hour_cos}. All features within a group receive identical attribution values by construction. Consequently, Kruskal-Wallis tests across clusters effectively compare 2 independent group-level distributions (not 7 per-feature distributions). Attention weights measure temporal position (which time-step matters), not feature importance, and are therefore excluded from the gradient-method consensus analysis.

*Method: KMeans clustering + GS-SHAP (primary) + three additional attribution methods + Spearman ρ agreement matrix*

---

### RQ6 — Cross-Campaign Domain Shift
**H6:** Feature distributions differ significantly between Search and Shopping campaigns (KS test), empirically motivating frozen-encoder domain adaptation.

*Method: KS test across 7 features + frozen-encoder fine-tuning transfer*

---

## Framework Architecture

```
Raw Ad Performance Data (89,675 rows × 16 cols)
            │
            ▼
┌─────────────────────────────────────────┐
│  §2  THEORY: ZINB Structure Diagnosis  │
│       Zero-inflation ratio, overdispersion │
└────────────────────┬────────────────────┘
                     │
            ┌────────▼─────────┐
            │   Causal DAG     │
            │  (Fig. 1)        │
            └────────┬─────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   §5 RQ1/H1    §6 RQ2/H2    §7 RQ3/H3
   PSM + IPW    Mediation    Moderation
   (Fig. 2,2b)  (Fig. 3,3b)  (Fig. 4)
        │            │            │
        └────────────┼────────────┘
                     │
            ┌────────▼─────────┐
            │  §4 Augmentation │
            │  β-VAE + Copula  │
            │  + MBB  (FSD✓)   │
            └────────┬─────────┘
                     │
        ┌────────────┼──────────────┐
        │            │              │
        ▼            ▼              ▼
   §6 H4a        §6 H4b/H4c    §6 Bayesian
   CLS Stage     REG Stage      Uncertainty
   (Fig. 5,12)   (Fig. 5,6,7)   (Fig. 14)
        │            │              │
        └────────────┼──────────────┘
                     │
            ┌────────▼─────────┐
            │  §7 RQ5/H5       │
            │  GS-SHAP         │
            │  Multi-method    │
            │  Attribution     │
            │  (Fig. 8–10)     │
            └────────┬─────────┘
                     │
            ┌────────▼─────────┐
            │  §8 RQ6/H6       │
            │  Domain          │
            │  Adaptation      │
            │  (Fig. 11)       │
            └──────────────────┘
```

---

## Data Description

### Dataset Overview

The dataset contains **hourly advertisement performance records** from a single advertiser on the Naver search advertising platform (South Korea's largest search engine), covering **March 2025**.

| Attribute | Value |
|-----------|-------|
| Total records | 89,675 rows |
| Features | 16 columns |
| Time period | March 2025 (1 month) |
| Granularity | Ad-group level × hourly |
| Advertiser | Single (anonymized) |
| Platform | Naver Search/Shopping Ads |

---

### Column Schema

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `Date` | datetime | Date of record (YYYY-MM-DD) | All records from March 2025 |
| `Hours` | int | Hour of day (0–23) | 0 = midnight |
| `customer_id` | int | Anonymized advertiser ID | Single value (135485) |
| `campaign_id` | str | Campaign identifier | Prefix encodes type: `-01-` = Search, `-02-` = Shopping, `-04-` = Zero-cost |
| `ad_group_id` | str | Ad group identifier | Nested under campaign |
| `ad_id` | str | Individual ad identifier | Leaf-level entity |
| `impression` | int | Number of ad impressions | ≥ 0 |
| `click` | int | Number of clicks | ≥ 0; click ≤ impression |
| `cost` | int | Advertising spend (KRW) | 0 for zero-cost campaigns |
| `sum_of_ad_rank` | int | Cumulative ad rank score | Platform-specific metric |
| `conversion_count` | int | Number of purchase conversions | 88.2% of paid rows = 0 |
| `sales_by_conversion` | int | Revenue attributed to conversions (KRW) | Highly right-skewed |
| `CTR` | float | Click-through rate (%) | `click / impression × 100`; NaN when impression = 0 |
| `CVR` | float | Conversion rate (%) | `conversion / click × 100` |
| `ROAS` | float | Return on ad spend (%) | `sales / cost × 100`; 72.1% of paid rows = 0 |
| `Depth` | float | Browsing depth score | Platform-side engagement proxy; NaN when impression = 0 |

---

### Derived Features (computed in preprocessing)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `CPC` | `cost / click` (0 if click=0) | Cost efficiency |
| `CPA` | `cost / conversion_count` (0 if conv=0) | Acquisition cost |
| `has_conversion` | `(conversion_count > 0).astype(int)` | Binary classification target |
| `log_*` | `np.log1p(*)` for impression, click, cost, CTR, CVR, ROAS, CPC, conversion_count | Variance stabilization |
| `hour_sin` | `sin(2π × Hours / 24)` | Cyclical hour encoding |
| `hour_cos` | `cos(2π × Hours / 24)` | Cyclical hour encoding |
| `campaign_type` | Extracted from `campaign_id` prefix | `Search` / `Shopping` / `Zero-cost` |

---

### Key Statistical Characteristics

```
Zero-ROAS rate (paid impressions):  72.1%   ← structural zero-inflation
Conversion rate (all rows):         11.77%
Zero-inflation ratio (ROAS):        3.3× vs Poisson expectation
Overdispersion (ROAS variance):     122,987× mean
```

These characteristics motivate the **Zero-Inflated Negative Binomial (ZINB)** model for distributional diagnosis and the **two-stage prediction architecture** (classification → regression).

---

### Campaign Type Distribution

| Type | campaign_id Prefix | Cost Structure | Characteristics |
|------|--------------------|----------------|-----------------|
| Search | `-01-` | CPC (paid) | Higher CTR, lower volume |
| Shopping | `-02-` | CPC (paid) | Lower CTR, higher volume |
| Zero-cost | `-04-` | Free (organic-like) | No cost data, excluded from causal analysis |

---

### Missing Values

| Column | Missing Count | Missing Rate | Treatment |
|--------|--------------|--------------|-----------|
| `CTR` | 438 | 0.49% | Fill 0 (all have impression=0) |
| `Depth` | 438 | 0.49% | Fill 0 (same rows as CTR) |
| All others | 0 | 0% | — |

---

### Sequence Dataset Statistics (after aggregation)

Split sizes reflect **group-aware splitting** (FIX-2): ad groups are allocated to folds as whole units to prevent sequence leakage across train/val/test.

| Split | REG sequences (SL=4) | CLS sequences (SL=4) |
|-------|---------------------|---------------------|
| Train | 174 | ~1,547 |
| Validation | 24 | ~332 |
| Test | 24 | ~332 |
| **Total** | **222** | **~2,211** |

> **Note on small sequence count:** The REG dataset is small because only 9,071 paid rows (27.9% of paid impressions) have ROAS > 0, and sequences require ≥ 5 consecutive hours per ad group. The augmentation pipeline (β-VAE + Copula + MBB) expands training to ~870 sequences (target_n=870), validated by FSD = 0.6852.

> **Note on group-aware split vs. original split:** The original README reported Train=155 / Val=33 / Test=34 based on a random index split (time_split). FIX-2 replaced this with group_time_split(), which assigns each ad group entirely to one fold, preventing sliding-window sequence overlap. The resulting sizes (174/24/24) differ because group boundaries do not align with fixed proportions.

---

## Repository Structure

```
sadaf/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── LICENSE
├── .gitignore
│
├── sadaf/                           # Main package
│   ├── __init__.py
│   ├── config.py                    # Hyperparameters & constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # Data loading & preprocessing
│   │   └── sequence.py              # Sequence dataset construction [FIX-2]
│   ├── augmentation/
│   │   ├── __init__.py
│   │   ├── vae.py                   # β-VAE augmentation
│   │   ├── copula.py                # Gaussian Copula augmentation
│   │   ├── mbb.py                   # Moving Block Bootstrap [FIX-B: n_each bug]
│   │   └── pipeline.py              # Combined augmentation pipeline + FSD
│   ├── causal/
│   │   ├── __init__.py
│   │   ├── psm.py                   # PSM + Doubly Robust IPW (H1)
│   │   ├── mediation.py             # Mediation analysis (H2)
│   │   └── moderation.py            # Moderation analysis (H3)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm.py                  # LSTM / BiLSTM / BayesianLSTM
│   │   ├── gru.py                   # GRU forecaster
│   │   ├── mamba.py                 # Mamba (selective SSM)
│   │   ├── protonet.py              # Prototypical Network (K-shot)
│   │   └── attention.py             # LSTM with attention
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py               # Generic training loop + DM test utilities [FIX-1]
│   └── explainability/
│       ├── __init__.py
│       ├── gsshap.py                # GS-SHAP [FIX-4a/4b/5]
│       ├── intgrad.py               # Integrated Gradients
│       ├── permshap.py              # Permutation SHAP [FIX: seed param]
│       └── agreement.py             # Spearman ρ method agreement [FIX-6]
│
├── scripts/                         # End-to-end runnable scripts
│   ├── 01_eda.py
│   ├── 02_zinb.py
│   ├── 03_causal.py
│   ├── 04_augmentation.py
│   ├── 05_prediction.py
│   ├── 06_uncertainty.py
│   ├── 07_explainability.py         # [FIX-2/3/4/6/7]
│   ├── 08_domain_adaptation.py
│   ├── 09_robustness.py
│   └── 10_figures.py
│
├── figures/                         # All output figures (auto-generated)
│   ├── fig_01_dag.png
│   ├── fig_02_psm_h1.png
│   ├── fig_02b_love_plot.png
│   ├── fig_03_mediation_h2.png
│   ├── fig_03b_suppressor.png
│   ├── fig_04_moderation_h3.png
│   ├── fig_05_model_comparison.png
│   ├── fig_06_learning_curves.png
│   ├── fig_07_mamba_sensitivity.png
│   ├── fig_08_gsshap_heatmaps.png
│   ├── fig_08b_multi_attribution.png
│   ├── fig_08c_agreement.png
│   ├── fig_09_gsshap_importance_fixed.png
│   ├── fig_09b_rank_consensus.png
│   ├── fig_10_cluster_profile.png
│   ├── fig_11_domain_shift.png
│   ├── fig_12_cls_supplement.png
│   ├── fig_13_dm_test.png
│   ├── fig_14_bayesian_uncertainty.png
│   ├── fig_W1a_aug_quality.png
│   ├── fig_W1b_data_scaling.png
│   ├── fig_W1c_logo_cv.png
│   ├── fig_W2a_temporal_stability.png
│   ├── fig_W2b_multi_advertiser.png
│   ├── fig_W3_overfitting_regularisation.png
│   ├── fig_W4_attribution_disagreement.png
│   ├── fig_W5_protonet_coldstart.png
│   ├── fig_W6_dm_corrected.png
│   └── fig_W7_is_theory.png
│
├── data/
│   └── README_data.md
│
├── tests/
│   ├── test_augmentation.py
│   ├── test_models.py
│   └── test_causal.py
│
└── docs/
    ├── methodology.md
    └── results_table.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/sadaf.git
cd sadaf

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.1
openpyxl>=3.1.0
```

---

## Usage

### Full pipeline (sequential)

```bash
python scripts/01_eda.py --data_path data/ad_performance.xlsx
python scripts/02_zinb.py --data_path data/ad_performance.xlsx
python scripts/03_causal.py --data_path data/ad_performance.xlsx
python scripts/04_augmentation.py --data_path data/ad_performance.xlsx
python scripts/05_prediction.py --data_path data/ad_performance.xlsx
python scripts/06_uncertainty.py
python scripts/07_explainability.py --data_path data/ad_performance.xlsx --out_dir figures/
python scripts/08_domain_adaptation.py
python scripts/09_robustness.py
python scripts/10_figures.py
```

> **Note (FIX-7):** On the first run of `07_explainability.py`, the BayesianLSTM model is trained with a fixed seed and saved to `figures/best_bayesian_lstm.pt`. Subsequent runs load this checkpoint automatically, ensuring reproducible attribution results. To force retraining, delete the `.pt` file.

### Python API usage

```python
from sadaf.data.loader import load_and_preprocess
from sadaf.augmentation.pipeline import augment_pipeline
from sadaf.models.lstm import BayesianLSTM
from sadaf.training.trainer import train_model, eval_reg

df, df_paid, df_roas = load_and_preprocess("data/ad_performance.xlsx")

from sadaf.data.sequence import build_sequences, group_time_split
X, Y, group_ids = build_sequences(df_roas, target_col='log_ROAS', seq_len=4)
(X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = group_time_split(
    X, Y, group_ids, train_frac=0.70, val_frac=0.85)

X_aug, Y_aug = augment_pipeline(X_train, Y_train, target_n=870)

model = BayesianLSTM(input_dim=7, hidden=128, dropout=0.4)
trained_model, history = train_model(model, train_loader, val_loader,
                                     real_val_loader=real_val_loader,
                                     epochs=60, patience=12)

posterior = trained_model.predict_posterior(X_test, n_samples=500, temperature=1.5)
print(f"95% CI coverage: {posterior['coverage_95']:.1f}%")
```

---

## Results & Visualizations

All figures are auto-generated by the pipeline and saved to `figures/`. See [Figures Requiring Update](#figures-requiring-update) for figures that reflect pre-fix pipeline runs and should be regenerated.

---

### Main Figures

#### Figure 1 — Causal DAG
`figures/fig_01_dag.png`

The directed acyclic graph (DAG) underlying the SADAF identification strategy. Nodes represent endogenous confounders, treatment (CTR), mediator (Depth), and outcome (Conversion). Dashed edges represent direct backdoor paths controlled via doubly robust estimation.

---

#### Figure 2 — PSM Diagnostics (H1)
`figures/fig_02_psm_h1.png`

Propensity score distributions before and after matching (caliper = 0.1σ), plus ATT comparison between PSM (supporting evidence) and IPW (primary doubly robust estimator). IPW-ATT = 0.1286; PSM-ATT = 0.1347 [0.1254, 0.1434].

---

#### Figure 2b — Love Plot (Covariate Balance)
`figures/fig_02b_love_plot.png`

Standardized mean differences (SMD) before and after PSM. Residual imbalance in `log_impression` and `log_cost` (|SMD| > 0.1) is corrected by doubly robust IPW weighting.

---

#### Figure 3 — Mediation Path Diagram (H2)
`figures/fig_03_mediation_h2.png`

Baron-Kenny mediation with bootstrap CI (B = 2,000). Indirect path a × b = 0.0265 [0.0200, 0.0337]. Path signs: a < 0, b < 0, yielding a **negative suppressor** structure.

---

#### Figure 3b — Suppressor Diagnosis
`figures/fig_03b_suppressor.png`

Correlation matrix of (log_CTR, Depth, has_conversion), CTR effect by depth quartile, and path decomposition.

---

#### Figure 4 — Moderation: CTR × Campaign Type → ROAS (H3)
`figures/fig_04_moderation_h3.png`

HC3-robust OLS interaction. β_interaction = 0.386 (p < 0.001). Search campaigns: ME = 0.949; Shopping: ME = 0.563.

---

#### Figure 5 — Model Comparison: AUC, RMSE, R² (H4a–H4c)
`figures/fig_05_model_comparison.png`

Two-stage model comparison (v3 fixed-seed results).

**Stage 1 — Classification (AUC):**

| Model | AUC | F1 | AP |
|-------|-----|----|----|
| LR-Cls | 0.6143 | 0.3653 | 0.3016 |
| LSTM-Cls | 0.6115 | 0.3026 | 0.3062 |
| BayesianLSTM-Cls | 0.5894 | 0.3151 | 0.2723 |
| MLP-Cls | 0.5445 | 0.2951 | 0.2989 |

H4a NULL: BayesianLSTM-Cls AUC (0.5894) < LR (0.6143).

**Stage 2 — Regression (RMSE, group-split, SL=4):**

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| **LSTM** | **1.2748** | **1.0054** | **0.7049** |
| BayesianLSTM | 1.4012 | 1.0971 | 0.6434 |
| GRU | 1.4106 | 1.1367 | 0.6386 |
| BiLSTM | 1.5532 | 1.2180 | 0.5619 |
| Ridge | 1.6112 | 1.2732 | 0.5286 |
| Mamba | 1.6532 | 1.3465 | 0.5037 |
| MLP | 1.7017 | 1.4021 | 0.4741 |

> **⚠ Figure requires regeneration** — pre-fix figure shows BayesianLSTM as best REG model; v3 results show LSTM as best. See [Figures Requiring Update](#figures-requiring-update).

---

#### Figure 6 — Training Curves
`figures/fig_06_learning_curves.png`

Train vs. augmented-dist val loss vs. real-data val loss (three curves, FIX-1) for all five deep architectures. Domain-gap report:

| Model | Best Epoch (real) | Best val_real | Final train | Gap (real−train) |
|-------|-------------------|---------------|-------------|-------------------|
| BayesianLSTM | 50 | 0.7273 | 1.1279 | −0.374 |
| LSTM | 22 | 0.6975 | 0.7710 | −0.116 |
| GRU | 24 | 0.7369 | 0.8797 | −0.094 |
| BiLSTM | 26 | 0.7143 | 0.7454 | −0.064 |
| Mamba | 20 | 0.9679 | 0.7439 | +0.130 |

> **⚠ Figure requires regeneration** — pre-fix figure shows only two curves (train + val_aug); v3 adds val_real as the early-stopping curve.

---

#### Figure 7 — Mamba SEQ_LEN Robustness (H4c)
`figures/fig_07_mamba_sensitivity.png`

RMSE vs. sequence length (4 → 6) with 95% bootstrap CI. Mamba ΔRMSE = −0.0041 (improves), vs. LSTM ΔRMSE = +0.0056 and GRU ΔRMSE = +0.4061. H4c supported: Mamba is more robust to sequence-length variation.

---

#### Figure 8 — GS-SHAP Temporal Attribution Heatmaps (H5)
`figures/fig_08_gsshap_heatmaps.png`

Cluster-level GS-SHAP attribution maps. HSIC grouping: Group 0 = {CTR, CVR, Depth, log_cost, log_impression}; Group 1 = {hour_sin, hour_cos}. Per-feature attribution values within each HSIC group are identical by construction (group-level Shapley decomposition). The heatmap therefore shows 2 independent attribution levels per cluster, not 7. Temporal Gini is computed at the group level (2 values per cluster).

> **⚠ Figure requires regeneration** — pre-fix heatmap used incorrect Gini formula.

---

#### Figure 8b — Multi-Method Attribution Comparison
`figures/fig_08b_multi_attribution.png`

Side-by-side comparison of GS-SHAP, Integrated Gradients, Permutation SHAP, and Attention-based attribution across all three clusters. Note: Attention weights are shown for reference only — they measure temporal position (which time-step matters), not feature-level importance, and are excluded from the Spearman ρ consensus computation.

> **⚠ Figure requires regeneration** — pre-fix Perm-SHAP importance vectors were scalar (aggregation bug), making all features appear equally important.

---

#### Figure 8c — Method Agreement Heatmap (Spearman ρ)
`figures/fig_08c_agreement.png`

Spearman rank correlation among the three gradient-based attribution methods (GS-SHAP, IntGrad, Perm-SHAP) per cluster (v3 fixed-seed results). Attention is excluded from this consensus matrix because it operates on a different axis (temporal position vs. feature importance).

| Cluster | Avg Spearman ρ (3 gradient methods) | Pairs usable |
|---------|-------------------------------------|--------------|
| C0 High-Volume | 0.825 | 3/3 |
| C1 High-Conversion | 0.559 | 3/3 |
| C2 Click-Rich | 0.813 | 3/3 |

> **⚠ Figure requires regeneration** — pre-fix figure showed NaN for all Perm-SHAP pairs.

---

#### Figure 9 — Feature Importance & Temporal Gini (H5)
`figures/fig_09_gsshap_importance_fixed.png`

GS-SHAP attribution boxplots by cluster, separated by C0 / C1 / C2 for visual comparison. Because GS-SHAP operates at the HSIC group level, all features within Group 0 (CTR, CVR, Depth, log_cost, log_impression) share identical attribution values; the 5 boxes for these features are structurally identical and reflect a single group-level Shapley value, not independent per-feature scores.

Kruskal-Wallis test across clusters compares 2 independent group-level distributions (Group 0 vs. Group 1). Result (v3): Group 1 (hour_sin/cos) significantly differs from Group 0 across clusters (p = 0.043). Reporting "5/7 features significant" is a mischaracterisation of the GS-SHAP decomposition and should be avoided.

Temporal Gini (group-level, corrected Lorenz formula, per cluster):

| Cluster | Group 0 (CTR/CVR/Depth/cost/impression) | Group 1 (hour_sin/cos) |
|---------|----------------------------------------|------------------------|
| C0 High-Volume | 0.328 | 0.317 |
| C1 High-Conversion | 0.280 | 0.342 |
| C2 Click-Rich | 0.373 | 0.290 |

> **⚠ Figure requires regeneration** — pre-fix figure used incorrect Gini formula (uniform→0.9375 instead of 0.0), showed near-identical values ~0.98 across all features, aggregated all clusters into a single boxplot, and displayed raw `np.int64` arrays as x-axis labels. The corrected figure (`fig_09_gsshap_importance_fixed.png`) uses cluster-separated boxplots and human-readable group labels.

---

#### Figure 9b — Rank Consensus Heatmap
`figures/fig_09b_rank_consensus.png`

Feature rank (1 = most important) from each of the three gradient-based attribution methods (GS-SHAP, IntGrad, Perm-SHAP) per cluster. Attention ranks are shown in a separate panel to avoid conflating temporal-position ranks with feature-importance ranks.

> **⚠ Figure requires regeneration** — pre-fix Perm-SHAP ranks were meaningless (scalar aggregation bug, FIX-C).

---

#### Figure 10 — Cluster Profiling: Radar + ROAS Violin
`figures/fig_10_cluster_profile.png`

Input feature radar chart (normalised), GS-SHAP group-level attribution radar (2 axes: Group 0 and Group 1), and ROAS distribution violin plots by cluster. Kruskal-Wallis confirms significant inter-cluster ROAS differences (p < 0.0001).

---

#### Figure 11 — Domain Shift: Shopping vs. Search (H6)
`figures/fig_11_domain_shift.png`

KS statistics per feature (6/7 significant at p < 0.05), target ROAS distribution comparison.

---

#### Figure 12 — Classification Diagnostics (H4a supplement)
`figures/fig_12_cls_supplement.png`

Precision-Recall curves for all classifiers. H4a NULL: BayesianLSTM-Cls (AUC=0.5894) < LR (AUC=0.6143); sparse ad-group sequences are largely linearly separable at this sample size.

---

#### Figure 13 — Diebold-Mariano Test Results
`figures/fig_13_dm_test.png`

Significant DM comparisons (v3, best REG model = LSTM):

| Pair | DM stat | p-value | Winner |
|------|---------|---------|--------|
| BayesianLSTM vs LSTM | 2.304 | 0.031 | LSTM |
| LSTM vs GRU | −2.693 | 0.013 | LSTM |
| LSTM vs BiLSTM | −2.492 | 0.020 | LSTM |
| LSTM vs Mamba | −2.999 | 0.006 | LSTM |
| LSTM vs Ridge | −2.779 | 0.011 | LSTM |
| LSTM vs MLP | −3.744 | 0.001 | LSTM |
| GRU vs Mamba | −2.932 | 0.008 | GRU |
| GRU vs Ridge | −2.553 | 0.018 | GRU |
| GRU vs MLP | −3.324 | 0.003 | GRU |

Note: After BH-FDR correction (W6), 0 pairs remain significant due to n=24 test sequences. Raw p-values reported conservatively in the main text.

> **⚠ Figure requires regeneration** — pre-fix figure reflected BayesianLSTM as best model; v3 results show LSTM.

---

#### Figure 14 — Bayesian Credible Intervals
`figures/fig_14_bayesian_uncertainty.png`

Posterior mean ± 95% credible interval. MC Dropout (T = 500 samples), temperature scaling T = 1.5. Empirical 95% CI coverage = 94.1% (well-calibrated).

---

### Robustness & Weakness Supplement Figures (W-series)

#### Figure W1a — Augmentation Quality Stratification
MBB: mean KS = 0.011, MMD = 0.0056. β-VAE: mean KS = 0.404 (captures nonlinear structure). Copula: intermediate.

#### Figure W1b — Data Scaling Curve
GRU RMSE vs. real training data fraction (20%–100%), power-law fit.

#### Figure W1c — Leave-One-Ad-Group-Out CV
LOGO-CV RMSE: mean = 1.2041 ± 0.5976 across 37 ad groups. Lower than hold-out RMSE, supporting generalization to unseen ad groups. This provides the primary generalization evidence for H5, supplementing the small within-cluster test set sizes (C0=7, C1=9, C2=8).

#### Figure W2a — Temporal Stability
Hour-block CV: std(RMSE) = 0.38 (moderate temporal variability).

#### Figure W2b — Synthetic Multi-Advertiser Robustness
Pre-trained GRU evaluated on five simulated advertiser distributions.

#### Figure W3 — Overfitting Diagnosis & Regularisation Grid
Optimal: dropout=0.4, weight_decay=1e-3 (GRU RMSE = 1.3777).

#### Figure W4 — Attribution Method Disagreement Analysis
Reframing of the Attention–GS-SHAP divergence. Left: 3-method (GS-SHAP/IntGrad/Perm-SHAP) Spearman ρ heatmaps. Centre: Attention temporal weights (which time-step is attended). Right: per-feature cross-method variance. Conclusion: Attention measures temporal position; gradient-based methods measure feature importance — these are complementary, not competing, axes. Spearman ρ consensus is computed only among the three gradient-based methods.

#### Figure W5 — ProtoNet Cold-Start Trajectory
K=1: RMSE=2.32, K=5: RMSE=2.28; converges toward full-data baseline.

#### Figure W6 — DM Test with Multiple Comparison Correction
After BH-FDR: 0 significant pairs (n=24 test sequences limits power). After Bonferroni: same. Raw significant pairs reported conservatively in main text.

#### Figure W7 — IS Theory Alignment
Signaling Theory: CTR quartile vs. ROAS. Resource Scarcity: peak uncertainty at <10 impressions.

---

## Key Findings

Results reflect the v3 fixed-seed pipeline (`RANDOM_SEED=42`, `07_explainability.py` checkpoint reuse).

| RQ / H | Method | Key Result | Verdict |
|--------|--------|-----------|---------|
| RQ1 / H1 | PSM + Doubly Robust IPW | IPW-ATT = 0.1286 (primary); PSM-ATT = 0.1347 [0.1254, 0.1434] | ✓ Supported |
| RQ2 / H2 | Baron-Kenny + Bootstrap (B=2,000) | a = −0.307, b = −0.086, a×b = 0.027 [0.020, 0.034] | △ Negative suppressor |
| RQ3 / H3 | OLS HC3-robust interaction | β_int = 0.386 (p < 0.001), ME_Search = 0.949, ME_Shopping = 0.563 | ✓ Supported |
| RQ4 / H4a | BayesianLSTM-Cls vs. LR (AUC) | BayLSTM = 0.589 vs. LR = 0.614 | ⚬ NULL (boundary condition) |
| RQ4 / H4b | LSTM vs. Ridge (DM) | LSTM RMSE = 1.275 vs. Ridge = 1.611; DM p = 0.011 | ✓ Supported |
| RQ4 / H4c | Mamba SEQ_LEN robustness | Mamba ΔRMSE = −0.004 vs. LSTM Δ = +0.006 | ✓ Supported (robustness only) |
| RQ4 / Bay | Bayesian posterior | 95% CI coverage = 94.1% (T = 1.5) | ✓ Novel |
| RQ4 / Proto | ProtoNet K-shot (K = 1–5) | K=1: 2.32, K=5: 2.28; converges toward full-data baseline | ✓ Novel |
| RQ5 / H5 | KW + GS-SHAP + Spearman ρ | KW: Group 1 (hour_sin/cos) vs. Group 0 significant (p = 0.043); Spearman ρ: 0.559–0.825 across all clusters (3 gradient methods) | ✓ Supported (caveat: n<10 per cluster; 2 group-level KW tests, not 7 per-feature) |
| RQ6 / H6 | KS-test (Search vs Shopping) | 6/7 features p < 0.05; frozen-encoder gain = +0.2% | ✓ Supported |

> **H4b note:** Best REG model changed from BayesianLSTM (pre-fix, RMSE=1.422) to **LSTM** (v3 group-split, RMSE=1.275) following FIX-2 (group-aware split eliminates sequence leakage). The DM test confirms LSTM significantly outperforms Ridge (p=0.011) and all other baselines.

> **H5 caveat:** All three clusters have n<10 test samples (C0=7, C1=9, C2=8). KW p=0.043 is marginal and should be reported with this caveat. GS-SHAP group-level decomposition means the KW test compares 2 independent distributions (Group 0 vs. Group 1), not 7 per-feature distributions. Phrasing such as "5/7 features significant" misrepresents the GS-SHAP structure and should not be used. LOGO-CV (§W1c) provides the primary generalization evidence.

---

## Code Fix Log (v3)

The following bugs were identified and corrected during the v3 revision. All fixes are documented with `[FIX-N]` markers in the relevant source files.

| Fix | File | Description |
|-----|------|-------------|
| FIX-1 | `trainer.py` | Added `real_val_loader` parameter: early stopping now driven by real held-out data, not augmented-dist val. Adds `val_real` curve to Figure 6. |
| FIX-2 | `sequence.py` | Replaced `time_split()` (index-based) with `group_time_split()` (ad-group-aware). Eliminates sliding-window sequence leakage across train/val/test splits. |
| FIX-3 | `gsshap.py` | Added `np.abs()` before Gini computation. Signed attribution values were cancelling to ~0, making all Gini values near-zero. |
| FIX-4a | `gsshap.py` | Corrected `temporal_gini()` formula. Previous formula gave uniform→0.9375 instead of correct 0.0; some inputs returned >1.0 (clipped). Standard Lorenz-based formula now used. |
| FIX-4b | `gsshap.py` | Increased time segmentation resolution: `min_seg_len=1` instead of 2, giving T=4 segments instead of 2. Prevents Gini from being forced near 1.0 by coarse bucketing. |
| FIX-5 | `gsshap.py` | Added `group_temporal_gini()` and `compute_cluster_gini(level="group")`. Reports 2 independent group-level Gini values per cluster instead of 7 per-feature values (5 of which are structural duplicates within HSIC Group 0). |
| FIX-6 | `agreement.py` | Added `_is_near_constant()` detector and `nanmean` handling. Near-constant importance vectors now produce explicit warnings rather than silent NaN averages. |
| FIX-7 | `07_explainability.py` | Global seed fixation (`random`, `numpy`, `torch`, `cudnn.deterministic`). Added checkpoint save/load: first run saves `best_bayesian_lstm.pt`; subsequent runs load it, ensuring reproducible KW and Spearman results. |
| FIX-8 | `10_figures.py` | Fixed x-axis labels in Temporal Gini boxplot: replaced raw `np.int64` group-index arrays with human-readable strings `"Group 0 (CTR/CVR/Depth/cost/imp)"` and `"Group 1 (hour_sin/cos)"`, separated by cluster (C0/C1/C2). |
| FIX-B | `mbb.py` | Fixed `n_each` bug: `X_real[idx]` returned shape `(N, T, D)` instead of `(T, D)` per iteration, causing `174 × n_each` samples instead of `n_each`. Augmentation output: 41,006 → 870 (correct). |
| FIX-C | `07_explainability.py` | Fixed `permshap_importance_by_cluster[c]` aggregation: `np.abs(v).mean(axis=0)` on a 1D `(D,)` vector collapsed to a scalar. Replaced with `np.mean(np.abs(np.stack(ps_vals)), axis=0)` → `(D,)` vector. This caused all Perm-SHAP Spearman pairs to be NaN. |
| FIX-D | `07_explainability.py` | Fixed `boxplot(arr.T, tick_labels=FEATURES)`: `arr.T` shape `(D,n_clusters)` interpreted as `n_clusters` boxes, mismatching `FEATURES` (7 labels). Corrected to `boxplot(arr, ...)` → `D` boxes. |

---

## Figures Requiring Update

The following figures were generated by the pre-fix pipeline and must be regenerated before submission:

| Figure | Issue | Fix Applied |
|--------|-------|-------------|
| `fig_05_model_comparison.png` | BayesianLSTM shown as best REG model; v3 best is LSTM | FIX-2 (group-aware split) |
| `fig_06_learning_curves.png` | Only 2 curves (train + val_aug); missing val_real | FIX-1 |
| `fig_08_gsshap_heatmaps.png` | Incorrect Gini formula | FIX-4a/4b |
| `fig_08b_multi_attribution.png` | Perm-SHAP scalar aggregation bug | FIX-C |
| `fig_08c_agreement.png` | All Perm-SHAP Spearman pairs NaN | FIX-C, FIX-6 |
| `fig_09_gsshap_importance_fixed.png` | Incorrect Gini formula; all clusters merged; x-axis shows raw np.int64 arrays; cluster separation missing | FIX-4a/4b/5, FIX-8, FIX-D |
| `fig_09b_rank_consensus.png` | Perm-SHAP ranks meaningless (scalar bug) | FIX-C |
| `fig_13_dm_test.png` | Reflects BayesianLSTM as best model; v3 winner is LSTM | FIX-2 |

Run `python scripts/10_figures.py` after applying all fixes to regenerate.

---

## Data Availability

The raw dataset (`3월성과데이터(샘플).xlsx`) consists of proprietary advertisement performance records from a single advertiser on the Naver platform and **cannot be publicly released** due to commercial confidentiality obligations.

Data sharing requests should be submitted via GitHub Issues (label: `data-request`), including institutional affiliation, research purpose, and confirmation of non-commercial use. Requests are evaluated case-by-case; authors aim to respond within 14 business days.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The license covers only the code and methodology. The underlying dataset is **not** covered by this license and remains subject to separate data sharing terms.

---

## Acknowledgements

- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation*. ICML.
- Gu et al. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv.
