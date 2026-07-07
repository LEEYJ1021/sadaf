# SADAF: Sparse Ad Data Augmentation Framework

> **A Unified Causal-Predictive-Explainable Framework for Cold-Start Advertisement Performance Forecasting**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 v5.1 Update Log (read this first)

This is a **reconciliation pass** on top of the v5 README. v5 was written
right after the FIX-9→FIX-23 pipeline run, before the full captured stdout
(`readme/README_v4_full.md`) had been cross-checked line by line. This
version fixes every place where v5 guessed or hedged, using
`readme/README_v4_full.md` as the source of truth.

**Treat the two files as a pair, not duplicates:**
- **`README.md` (this file)** — curated narrative, headline tables, verdicts.
- **`readme/README_v4_full.md`** — the raw captured stdout of the entire
  `01_eda.py → 09_robustness.py` run. Any number in this README should be
  traceable to a line in that file. If they ever disagree, the full log
  wins and this file needs a follow-up patch.

**What changed in this pass:**

| # | Issue in the v5 README | Resolution |
|---|--------------------------|------------|
| 1 | Figure 5's LSTM row was flagged as "truncated in captured stdout, re-run needed" | **Not actually truncated.** `readme/README_v4_full.md` → §4 → `Table 2b` has the full row: `LSTM 1.2099 0.9608 0.7342`. Table below is now final, no re-run needed. |
| 2 | FSD reported as a single "run-to-run varies" number (0.6852 / −0.1347 / −1.0057) with no explanation | The full log shows **two different FSD values from two different call sites**, not just noise: `05_prediction.py`/`07_explainability.py` augment from N_train=174 → target_n=870 and get **FSD=−0.0465**; `09_robustness.py`'s LOGO-CV context augments from N_train=155 → target_n≈800 and gets **FSD=−1.0057**. Both pass (<2.0), but they are not the same experiment — see §Appendix W note below. |
| 3 | H4a verdict framing | The hypothesis (H4a) is specifically about **BayesianLSTM-Cls vs LR**. The pipeline's own printed verdict line compares **LSTM-Cls vs LR-Cls** instead. Both comparisons reach the same conclusion (NULL — LR wins), but this is a framing mismatch between the hypothesis and the script's verdict logic, noted explicitly below rather than silently smoothed over. |
| 4 | *(new, found in this pass)* Appendix W6's DM-with-multiple-comparison-correction table reports **different p-values for the same model pairs** already reported in Figure 13's raw+FDR table (e.g. LSTM vs Mamba: p_raw=0.0024 in Fig.13 vs p=0.0199 in W6) | **Not reconciled yet.** Flagged as an open item in Appendix W below — the two tables appear to come from independent DM computations (possibly different resampling/subset), and should not be merged or averaged until the source of the discrepancy is identified in code. |

---

## Table of Contents
1. Overview
2. Research Questions (v5 — Korea Case-Study Framing)
3. Framework Architecture
4. Data Description
5. Repository Structure
6. Installation
7. Usage
8. Results & Visualizations
9. Key Findings (v5.1, reconciled)
10. Code Fix Log (v3 + v5)
11. Open Items / Figures Requiring Update
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

A custom three-method augmentation pipeline (β-VAE + Gaussian Copula + Moving Block Bootstrap) addresses the fundamental data scarcity problem.

> **Note on FSD (reconciled, v5.1):** There is no single canonical FSD
> value for this pipeline — there are (at least) two, from two different
> call sites, and both are legitimate:
> - `05_prediction.py` / `07_explainability.py`: N_train=174 → augmented
>   to target_n=870 → **FSD = −0.0465**
> - `09_robustness.py` (LOGO-CV context, one fold held out): N_train=155
>   → augmented to target_n≈800 → **FSD = −1.0057**
>
> Both are well inside the <2.0 acceptance threshold. The v3 README's
> 0.6852 and the superseded −0.1347 come from an earlier pre-FIX-21 run
> and are no longer reproducible as stated — do not cite them going
> forward. See the open item in **§Appendix W / Open Items** below on why
> exact FSD values still aren't reproducible **within** either call site
> across separate runs (Copula/MBB don't yet take an explicit seed).

> **Why Korea, why March 2025, why one advertiser?** Framed explicitly as
> an intentional **boundary-condition case study**, not an incidental
> limitation. See §2, RQ0 below.

---

## Research Questions (v5 — Korea Case-Study Framing)

SADAF v5 keeps the original six research questions (RQ1–RQ6) unchanged in
method, but wraps them in an explicit scope statement (RQ0) and adds two
new questions (RQ4d, RQ7) that were previously implicit.

### RQ0 — Scope framing (not a tested hypothesis) `[v5]`
Do causal, predictive, and explainability patterns established primarily
in Google-dominated advertising markets replicate under a structurally
different, single-platform-concentrated search ecosystem? March 2025
Korea (Naver ≈ 63.8% search share) is treated as a natural
boundary-condition test case, not as a claim of representativeness for
the Korean market as a whole.

> According to InternetTrend data reported by BusinessKorea (Apr 2026),
> Naver held an average **63.8%** share of Korean search volume in March
> 2025 versus Google's **28.7%** (Feb 2025: Naver 65.1%) — a
> platform-concentration structure with no equivalent in the
> Google-dominated markets (>90% Google share) that most cold-start /
> computational-advertising literature is built on.
> *Source (verify before submission): BusinessKorea, "Naver's Search
> Market Share Hits 64% While Google Ranked 2nd with 29% Share" (Apr 16,
> 2026), citing InternetTrend monthly tracking data.*

**This advertiser's descriptive scope indicators:**

| Metric | Value |
|--------|-------|
| Campaign-type mix | Shopping 78.83% (70,693 rows) / Search 19.37% (17,373 rows) / Zero-cost 1.79% (1,609 rows) |
| Ad-group spend concentration (HHI, 0–10,000 scale) | 201.0 |
| Top-3 spend-share hours (KST) | Hour 0: 77.65% · Hour 1: 9.47% · Hour 3: 2.33% |

---

### RQ1 — Causal Effect of CTR on Conversion
**H1:** High click-through rate (CTR) ads causally increase conversion probability compared to low-CTR ads, after controlling for impression volume, cost, and campaign type.

*Method: Propensity Score Matching (PSM, caliper = 0.1σ) + Doubly Robust Inverse Probability Weighting (IPW) — `[FIX-11]`*

---

### RQ2 — Mediating Role of Browsing Depth
**H2:** Browsing depth (Depth) mediates the relationship between CTR and conversion, and the sign pattern of the mediation paths reveals structural characteristics of advertiser-consumer interaction.

*Method: Baron-Kenny decomposition + Bootstrap mediation (B = 2,000) — `[FIX-12]`*

---

### RQ3 — Campaign-Type Moderation
**H3:** The positive relationship between CTR and ROAS is moderated by campaign type, such that Search campaigns exhibit a stronger CTR→ROAS slope than Shopping campaigns.

*Method: OLS interaction with HC3-robust standard errors — `[FIX-13]`*

---

### RQ4 — Deep Sequential ROAS Prediction
**H4a:** A Bayesian LSTM classifier outperforms logistic regression for binary ROAS prediction (zero vs. non-zero) on augmented sparse ad sequences.

> **`[v5.1 note]`** The pipeline's printed H4a verdict actually compares
> **LSTM-Cls vs LR-Cls**, not BayesianLSTM-Cls vs LR-Cls as the hypothesis
> states. Both framings currently reach the same verdict (NULL — LR wins
> both), so this does not change the conclusion, but the script's verdict
> logic should be updated to test the hypothesis as written. See Table 2a
> in §8 below for both models' numbers side by side.

**H4b:** The best-performing recurrent architecture achieves significantly lower RMSE than ridge regression and MLP baselines for log-ROAS prediction, as confirmed by Diebold-Mariano (DM) tests.

**H4c:** Mamba (selective state-space model) exhibits greater robustness to sequence-length variation (SEQ_LEN 4 → 6) compared to standard LSTM and GRU, measured by ΔRMSE per +2 time steps.

**H4d — Domain gap as diagnostic evidence** `[v5]`**:** Does the augmentation-to-real domain gap itself differ systematically across architectures in a way diagnostic of overfitting risk under extreme cold-start sparsity (N_train=174 real sequences)? Reported explicitly rather than minimized.

*Method: BayesianLSTM / GRU / BiLSTM / Mamba trained on β-VAE + Copula + MBB augmented data; DM test with raw + BH-FDR corrected p-values — `[FIX-10, FIX-22, FIX-23]`*

---

### RQ5 — Cluster-Specific Attribution Explanation
**H5:** Ad group clusters exhibit statistically distinct feature attribution patterns across HSIC-defined feature groups (measured by Kruskal-Wallis η²), and multiple attribution methods (GS-SHAP, Integrated Gradients, Permutation-SHAP) produce convergent rankings.

> **Note on GS-SHAP group structure:** GS-SHAP decomposes attribution at
> the HSIC group level, not the individual feature level. Group 0 =
> {CTR, CVR, Depth, log_cost, log_impression}; Group 1 = {hour_sin,
> hour_cos}. All features within a group receive identical attribution
> values by construction. **`[FIX-9]`** the Kruskal-Wallis test — and
> therefore the H5 verdict — is computed and reported over exactly these
> **2 independent group-level distributions**, never over "7 features."
> Attention weights measure temporal position (which time-step matters),
> not feature importance, and remain excluded from the gradient-method
> consensus analysis.

*Method: KMeans clustering + GS-SHAP (primary) + three additional attribution methods + Spearman ρ agreement matrix*

---

### RQ6 — Cross-Campaign Domain Shift
**H6:** Feature distributions differ significantly between Search and Shopping campaigns (KS test), empirically motivating frozen-encoder domain adaptation.

*Method: KS test across 7 features + frozen-encoder fine-tuning transfer — `[FIX-14, FIX-15]`*

---

### RQ7 — External-validity boundary `[v5]`
LOGO-CV (Appendix W1c) and the synthetic multi-advertiser check (Appendix
W2b) support generalization **within** this single-platform,
single-month case. Claims beyond that scope (other platforms, other
months, other advertisers) are explicitly out of scope for this study.

---

## Framework Architecture

```
Raw Ad Performance Data (89,675 rows × 32 cols)
            │
            ▼
┌─────────────────────────────────────────┐
│  §2  THEORY: ZINB Structure Diagnosis  │
│       AIC=71958.2  ΔAIC(ZIP−ZINB)=-2798.9 │
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
   [FIX-11]     [FIX-12]     [FIX-13]
        │            │            │
        └────────────┼────────────┘
                     │
            ┌────────▼─────────┐
            │  §4 Augmentation │
            │  β-VAE + Copula  │
            │  + MBB  [FIX-21] │
            │  FSD=-0.0465 or  │
            │  FSD=-1.0057*    │
            └────────┬─────────┘
                     │
        ┌────────────┼──────────────┐
        │            │              │
        ▼            ▼              ▼
   §6 H4a        §6 H4b/H4c    §6 Bayesian
   CLS Stage     REG Stage      Uncertainty
   [FIX-10/22/23] Winner: LSTM  (Fig. 14)
   LR wins ✗      RMSE=1.2099
        │            │              │
        └────────────┼──────────────┘
                     │
            ┌────────▼─────────┐
            │  §7 RQ5/H5       │
            │  GS-SHAP         │
            │  [FIX-9]         │
            │  1/2 groups sig. │
            └────────┬─────────┘
                     │
            ┌────────▼─────────┐
            │  §8 RQ6/H6       │
            │  Domain          │
            │  Adaptation      │
            │  [FIX-14/15]     │
            │  6/7 KS sig.     │
            └──────────────────┘

  * two different FSD values from two different call sites — see FSD note above
```

---

## Data Description

### Dataset Overview

The dataset contains **hourly advertisement performance records** from a single advertiser on the Naver search advertising platform (South Korea's largest search engine), covering **March 2025**.

| Attribute | Value |
|-----------|-------|
| Total records | 89,675 rows |
| Features (raw + derived, confirmed from `01_eda.py`) | 32 columns |
| Time period | March 2025 (1 month) |
| Granularity | Ad-group level × hourly |
| Advertiser | Single (anonymized) |
| Platform | Naver Search/Shopping Ads |

---

### Column Schema (raw source columns)

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

### Derived Features (confirmed present in the loaded DataFrame, 32 total columns)

`CPC`, `CPA`, `has_conversion`, `log_impression`, `log_click`, `log_cost`,
`log_CTR`, `log_CVR`, `log_ROAS`, `log_CPC`, `log_conversion_count`,
`hour_sin`, `hour_cos`, `campaign_type`, `campaign_type_label`, `hour_bin`.

> `campaign_type_label` is the human-readable column (`Search` /
> `Shopping` / `Zero-cost`) — use this one, not the raw `campaign_type`
> code, for any market-context reporting.

---

### Key Statistical Characteristics (confirmed, `01_eda.py` + `02_zinb.py`)

```
Total rows                :  89,675
Paid rows                 :  32,494
ROAS > 0                  :   9,071  (27.9% of paid)
Conversion rate           :  11.77%
Zero-ROAS rate (paid)     :  72.1%   ← structural zero-inflation
ROAS mean                 :  1852.37
ROAS variance             :  236,661,259.29
Overdispersion (var/mean) :  127,761.02
```

**ZINB model fit (confirmed):**

| Metric | Value |
|--------|-------|
| AIC | 71,958.2 |
| BIC | 72,025.3 |
| ΔAIC (ZIP − ZINB) | −2,798.9 (ZINB strongly preferred, threshold >10) |
| Convergence | lbfgs, standard errors valid |

Selected ZINB coefficients (count component): `log_CTR` = 0.4725
(p<0.001), `log_cost` = −0.2157 (p<0.001), `log_impression` = 0.2184
(p<0.001). Inflation component: `inflate_log_CTR` = −0.1895 (p<0.001),
`inflate_log_cost` = −0.5814 (p<0.001). Full coefficient table with
standard errors is in `readme/README_v4_full.md` → Appendix B.

These characteristics motivate the **Zero-Inflated Negative Binomial (ZINB)** model for distributional diagnosis and the **two-stage prediction architecture** (classification → regression).

---

### Campaign Type Distribution (confirmed, this advertiser)

| Type | campaign_id Prefix | Rows | Share |
|------|--------------------|------|-------|
| Shopping | `-02-` | 70,693 | 78.83% |
| Search | `-01-` | 17,373 | 19.37% |
| Zero-cost | `-04-` | 1,609 | 1.79% |

---

### Missing Values

| Column | Missing Count | Missing Rate | Treatment |
|--------|--------------|--------------|-----------|
| `CTR` | 438 | 0.49% | Fill 0 (all have impression=0) |
| `Depth` | 438 | 0.49% | Fill 0 (same rows as CTR) |
| All others | 0 | 0% | — |

> **`[v5.1 note]`** `01_eda.py`'s own missing-value check prints "결측값
> 없음" (no missing values) against the raw 32-column DataFrame, while
> this table (carried over from earlier versions) reports 438 missing
> `CTR`/`Depth` values. These are not contradictory — the raw check runs
> **before** the `CTR`/`Depth` fill-with-0 step in preprocessing, and
> "no missing values" describes the *post-fill* state used everywhere
> else. Kept as-is for clarity on the imputation itself.

---

### Sequence Dataset Statistics (group-aware split, confirmed)

| Split | REG sequences (SL=4) |
|-------|---------------------|
| Train | 174 |
| Validation | 24 |
| Test | 24 |
| **Total** | **222** |

Additional sequence-length variant used for H4c robustness: **SEQ_LEN=6 → (125, 6, 7)**.

> **Augmentation scale (confirmed, two distinct runs):**
> - `05_prediction.py` / `07_explainability.py`: 174 → ~870 sequences
>   (+232 per method: β-VAE + Copula + MBB), **FSD = −0.0465**
> - `09_robustness.py` LOGO-CV context: 155 → ~800 sequences (+215 per
>   method), **FSD = −1.0057**
>
> The N_train difference (174 vs 155) reflects that the LOGO-CV context
> holds one ad-group fold out before augmenting, so it always trains on
> slightly fewer real sequences than the main train split.

---

## Repository Structure

```
sadaf/
│
├── README.md                        # This file (curated narrative)
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── sadaf/
│   ├── config.py                    # RANDOM_SEED, DEVICE, hyperparameters
│   ├── data/
│   │   ├── loader.py
│   │   └── sequence.py              # build_sequences(), group_time_split(), SeqDataset
│   ├── augmentation/
│   │   ├── vae.py
│   │   ├── copula.py                # ⚠ no explicit seed param yet (see Open Items)
│   │   ├── mbb.py                   # ⚠ no explicit seed param yet (see Open Items)
│   │   └── pipeline.py              # [FIX-21] seed refixation in train_vae()/vae_augment()
│   ├── causal/
│   │   ├── psm.py                   # run_psm_ipw() [FIX-11]
│   │   ├── mediation.py             # run_mediation() [FIX-12]
│   │   └── moderation.py            # run_moderation() [FIX-13]
│   ├── models/
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   ├── mamba.py
│   │   ├── protonet.py
│   │   └── attention.py
│   ├── training/
│   │   └── trainer.py               # train_model, eval_reg, diebold_mariano
│   └── explainability/
│       ├── gsshap.py                # group_temporal_gini(), compute_cluster_gini(level="group")
│       ├── intgrad.py
│       ├── permshap.py
│       └── agreement.py
│
├── scripts/
│   ├── 01_eda.py
│   ├── 02_zinb.py
│   ├── 03_causal.py                 # [FIX-11/12/13]
│   ├── 04_augmentation.py
│   ├── 05_prediction.py             # [FIX-10/22/23]
│   ├── 06_uncertainty.py
│   ├── 07_explainability.py         # [FIX-9], requires --out_dir
│   ├── 08_domain_adaptation.py      # [FIX-14/15]
│   ├── 09_robustness.py             # [FIX-16/17/18/19/19b/20]
│   └── 10_figures.py
│
├── readme/
│   └── README_v4_full.md            # Full captured stdout log — source of truth for exact numbers
│
├── figures/                          # All output figures (auto-generated)
│   └── best_bayesian_lstm.pt        # BayesianLSTM checkpoint (FIX-7); delete to force retrain
├── data/
│   └── README_data.md
├── tests/
└── docs/
```

---

## Installation

```bash
git clone https://github.com/LEEYJ1021/sadaf.git
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

> `07_explainability.py` requires `--out_dir` as a mandatory argument
> (this is where `best_bayesian_lstm.pt` and Figure 9 are saved). On the
> first run, the BayesianLSTM model is trained with a fixed seed and
> checkpointed; subsequent runs reload it for reproducible attribution
> results. Delete the `.pt` file to force retraining.

---

## Results & Visualizations

All figures are auto-generated by the pipeline and saved to `figures/`.
Every number below is confirmed against `readme/README_v4_full.md`.

### Table 2a — Classification Stage (H4a)

| Model | AUC | F1 | AP |
|-------|-----|----|----|
| LR-Cls | **0.6143** | 0.3653 | 0.3016 |
| LSTM-Cls | 0.6115 | 0.3026 | 0.3062 |
| BayesianLSTM-Cls | 0.5894 | 0.3151 | 0.2723 |
| MLP-Cls | 0.5445 | 0.2951 | 0.2989 |

**H4a: NULL ⚬** — LR-Cls has the highest AUC. This holds whether you
compare against BayesianLSTM-Cls (the hypothesis as written: 0.5894 <
0.6143) or LSTM-Cls (the script's printed verdict: 0.6115 < 0.6143).
Interpretation: sparse ad-group sequences at this sample size are largely
linearly separable, so recurrent classifiers gain nothing over logistic
regression for the binary has-conversion task.

---

### Table 2b — Regression Stage (H4b/H4c), confirmed final

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| **LSTM** | **1.2099** | **0.9608** | **0.7342** |
| BayesianLSTM | 1.3420 | 1.1063 | 0.6729 |
| GRU | 1.3984 | 1.1450 | 0.6449 |
| BiLSTM | 1.4998 | 1.1629 | 0.5915 |
| Ridge | 1.6033 | 1.2684 | 0.5331 |
| Mamba | 1.6356 | 1.3308 | 0.5142 |
| MLP | 1.7086 | 1.3538 | 0.4699 |

**H4b: SUPPORTED ✓** — LSTM (RMSE=1.2099) significantly beats Ridge
(RMSE=1.6033); confirmed by DM test below (p_raw=0.0078, p_FDR=0.0316).

**Domain-gap report (train–val gap, supports H4d):**

| Model | Best epoch (real) | Best val_real | Best val_aug | Final train | gap_real | gap_aug |
|-------|---------------------|-----------------|-----------------|----------------|----------|---------|
| BayesianLSTM | 42 | 0.7525 | 0.7441 | 1.0936 | −0.2821 | −0.3494 |
| LSTM | 22 | 0.7047 | 0.7133 | 0.8080 | −0.0862 | −0.0947 |
| GRU | 36 | 0.6873 | 0.7131 | 0.8312 | −0.1591 | −0.1181 |
| BiLSTM | 23 | 0.7587 | 0.6964 | 0.8477 | −0.0802 | −0.1513 |
| Mamba | 35 | 0.7910 | 0.4564 | 0.6138 | +0.1142 | −0.1574 |

BayesianLSTM shows by far the largest real/train gap (most overfitting
risk under this sparsity); Mamba is the only model with a *positive*
gap_real (final train loss slightly worse than best real-val loss),
consistent with its comparatively low R² and its role as the
robustness-focused architecture (H4c) rather than the raw-accuracy
winner.

---

### Figure 13 — Diebold-Mariano Test Results (primary, raw + BH-FDR — `[FIX-10]`)

| Pair | DM stat | p_raw | p_FDR | Winner |
|------|---------|-------|-------|--------|
| BayesianLSTM vs LSTM | 3.1782 | 0.0042* | 0.0293* | LSTM |
| BayesianLSTM vs GRU | −1.1323 | 0.2692 | 0.3140 | BayesianLSTM |
| BayesianLSTM vs BiLSTM | −1.6517 | 0.1122 | 0.1472 | BayesianLSTM |
| BayesianLSTM vs Mamba | −2.3879 | 0.0255* | 0.0537 | BayesianLSTM |
| BayesianLSTM vs Ridge | −2.1653 | 0.0410* | 0.0615 | BayesianLSTM |
| BayesianLSTM vs MLP | −2.9221 | 0.0077* | 0.0316* | BayesianLSTM |
| LSTM vs GRU | −2.7851 | 0.0105* | 0.0316* | LSTM |
| LSTM vs BiLSTM | −2.7881 | 0.0105* | 0.0316* | LSTM |
| LSTM vs Mamba | −3.4113 | 0.0024* | 0.0251* | LSTM |
| LSTM vs Ridge | −2.9124 | 0.0078* | 0.0316* | LSTM |
| LSTM vs MLP | −3.4587 | 0.0021* | 0.0251* | LSTM |
| GRU vs BiLSTM | −1.5906 | 0.1254 | 0.1549 | GRU |
| GRU vs Mamba | −2.3405 | 0.0283* | 0.0540 | GRU |
| GRU vs Ridge | −2.4195 | 0.0239* | 0.0537 | GRU |
| GRU vs MLP | −2.7088 | 0.0125* | 0.0329* | GRU |
| BiLSTM vs Mamba | −2.2002 | 0.0381* | 0.0615 | BiLSTM |
| BiLSTM vs Ridge | −2.2157 | 0.0369* | 0.0615 | BiLSTM |
| BiLSTM vs MLP | −2.0115 | 0.0561 | 0.0786 | BiLSTM |
| Mamba vs Ridge | 0.4228 | 0.6763 | 0.6763 | Ridge |
| Mamba vs MLP | −0.5872 | 0.5628 | 0.5909 | Mamba |
| Ridge vs MLP | −0.9403 | 0.3568 | 0.3944 | Ridge |

**Summary: 14/21 pairs significant at raw p<0.05; 8/21 remain significant
after BH-FDR correction** (test sequences n=24).

> ⚠ **See Appendix W / Open Items below** — a second, independently
> computed DM table (Appendix W6) reports different p-values for these
> same pairs. Use **this** table (Figure 13 / FIX-10) as primary until
> the discrepancy is resolved.

---

### Figure 9 — GS-SHAP Group-Level Attribution & Temporal Gini (`[FIX-9]`)

| Cluster | Group 0 (CTR/CVR/Depth/log_cost/log_impression) | Group 1 (hour_sin/hour_cos) |
|---------|--------------------------------------------------|------------------------------|
| C0 High-Volume (n=7) | 0.328 | 0.317 |
| C1 High-Conversion (n=9) | 0.280 | 0.342 |
| C2 Click-Rich (n=8) | 0.373 | 0.290 |

**Kruskal-Wallis (group-level):**

| Group | p-value | Significance |
|-------|---------|---------------|
| Group 0 | 0.0428 | * |
| Group 1 | 0.688 | ns |

**H5 [FIX-9]: SUPPORTED ✓** — 1/2 HSIC group-level tests significant.
Explicitly **not** a "5/7 features significant" statement.

**Method agreement (Spearman ρ, gradient methods only):**

| Cluster | Avg Spearman ρ | n (test samples) |
|---------|-----------------|-------------------|
| C0 High-Volume | 0.825 (High) | 7 ⚠ underpowered |
| C1 High-Conversion | 0.559 (Moderate) | 9 ⚠ underpowered |
| C2 Click-Rich | 0.813 (High) | 8 ⚠ underpowered |

All three clusters have n<10 test samples; agreement and KW statistics
should be reported with this caveat. LOGO-CV (§Appendix W1c) supplies the
primary generalization evidence for H5.

---

### §H6 — Domain Shift

| Feature | KS statistic | p-value | Significant? |
|---------|--------------|---------|----------------|
| CTR | 0.2200 | 1.51e-81 | * |
| CVR | 0.0383 | 7.10e-03 | * |
| Depth | 0.3749 | 1.13e-239 | * |
| log_cost | 0.2271 | 6.41e-87 | * |
| log_impression | 0.1310 | 4.94e-29 | * |
| hour_sin | 0.0940 | 3.72e-15 | * |
| hour_cos | 0.0296 | 6.79e-02 | ns |

**H6: SUPPORTED ✓** (6/7 features p<0.05)

**Domain adaptation (Search → Shopping, frozen-encoder fine-tuning):**

| Transfer setup | RMSE | Gain vs. naive |
|-----------------|------|------------------|
| Naive transfer | 1.3133 | — |
| Adapted (50% frozen) | 1.3133 | −0.0% |

> No measurable improvement in this run; the primary contribution of
> this analysis is the theoretical justification for domain-adaptive
> design (motivated by the KS-test result above), not a performance
> claim.

---

## Key Findings (v5.1, reconciled)

| RQ / H | Method | Key Result (confirmed) | Verdict |
|--------|--------|---------------------------|---------|
| RQ1 / H1 | PSM + Doubly Robust IPW `[FIX-11]` | IPW-ATT = 0.1286 (primary); PSM-ATT = 0.1347 [0.1254, 0.1434]; n_matched = 14,987; DR consistent | ✓ Supported |
| RQ2 / H2 | Baron-Kenny + Bootstrap (B=2,000) `[FIX-12]` | a = −0.3077, b = −0.0861, a×b = 0.0265 [0.0200, 0.0337] | ✓ Supported (negative suppressor) |
| RQ3 / H3 | OLS HC3-robust interaction `[FIX-13]` | β_int = 0.3860 (p < 0.0001); ME_Search = 0.949; ME_Shopping = 0.563; R² = 0.6551, n = 9,069 | ✓ Supported |
| RQ4 / H4a | LR-Cls vs. LSTM-Cls / BayesianLSTM-Cls | LR-Cls AUC=0.6143 beats both (LSTM-Cls 0.6115, BayesianLSTM-Cls 0.5894) | ⚬ NULL (boundary condition) |
| RQ4 / H4b | LSTM vs. Ridge (DM, raw + FDR) `[FIX-10/22/23]` | **LSTM RMSE = 1.2099** vs. Ridge RMSE = 1.6033; DM p_raw = 0.0078, p_FDR = 0.0316 | ✓ Supported |
| RQ4 / H4d | Domain-gap report | BayesianLSTM largest real/train gap (−0.28 to −0.35); Mamba only model with positive gap_real | ✓ Novel (reported, not minimized) |
| RQ5 / H5 | KW (group-level) + GS-SHAP + Spearman ρ `[FIX-9]` | Group 0 vs. Group 1 significant (p = 0.0428); Spearman ρ 0.559–0.825 across clusters | ✓ Supported (caveat: n<10 per cluster) |
| RQ6 / H6 | KS-test (Search vs Shopping) `[FIX-14/15]` | 6/7 features p < 0.05; frozen-encoder adaptation gain ≈ 0% | ✓ Supported (H6); adaptation benefit not demonstrated |
| RQ7 | LOGO-CV + multi-advertiser | LOGO-CV RMSE = 1.2427 ± 0.6042 (n_groups = 37) | ✓ Supports within-scope generalization |

> **H5 caveat (unchanged):** All three clusters have n<10 test samples
> (C0=7, C1=9, C2=8). The group-level KW p=0.0428 is marginal. Phrasing
> such as "5/7 features significant" misrepresents the GS-SHAP
> group-level decomposition and must not be used.

---

## Code Fix Log (v3 + v5)

### v3 fixes (unchanged from prior documentation)

| Fix | File | Description |
|-----|------|-------------|
| FIX-1 | `trainer.py` | Added `real_val_loader`; early stopping driven by real held-out data. |
| FIX-2 | `sequence.py` | `group_time_split()` replaces index-based `time_split()`. |
| FIX-3 | `gsshap.py` | Added `np.abs()` before Gini computation. |
| FIX-4a/4b | `gsshap.py` | Corrected `temporal_gini()` Lorenz formula; increased segmentation resolution. |
| FIX-5 | `gsshap.py` | Added group-level Gini reporting (2 values, not 7 duplicates). |
| FIX-6 | `agreement.py` | `_is_near_constant()` detector; explicit warnings instead of silent NaN. |
| FIX-7 | `07_explainability.py` | Global seed fixation + BayesianLSTM checkpoint save/load. |
| FIX-8 | `10_figures.py` | Human-readable x-axis labels, separated by cluster. |
| FIX-B | `mbb.py` | Fixed `n_each` bug (41,006 → 870). |
| FIX-C | `07_explainability.py` | Fixed Perm-SHAP aggregation collapsing to a scalar. |
| FIX-D | `07_explainability.py` | Fixed `boxplot(arr.T, ...)` shape mismatch. |

### v5 fixes

| Fix | File | Description |
|-----|------|-------------|
| FIX-9 | `07_explainability.py` | H5 Kruskal-Wallis at HSIC group level (2 tests), not per-feature (7). |
| FIX-10 | `05_prediction.py` | DM test reports raw p **and** BH-FDR corrected p for every pair. |
| FIX-11/12/13 | `03_causal.py` + `sadaf/causal/*` | `run_h1/h2/h3()` updated to current single-function APIs. |
| FIX-14/16 | `08_domain_adaptation.py`, `09_robustness.py` | `SeqDataset` import path corrected to `sadaf.data.sequence`. |
| FIX-15/17 | `08_domain_adaptation.py`, `09_robustness.py` | `build_sequences()` keyword/arity corrected. |
| FIX-18 | `09_robustness.py` | `augment_pipeline(ref_model=...)` keyword corrected. |
| FIX-19/19b/20 | `09_robustness.py` | Models + batches moved to `DEVICE`; missing `DEVICE` import added. |
| FIX-21 | `sadaf/augmentation/pipeline.py` | `train_vae()` / `vae_augment()` re-fix RNG seed on entry. |
| FIX-22/23 | `05_prediction.py` | Seed re-fixed before `model_registry` instantiation and before each model's training loop. |

All fixes are idempotent: each checks for its own `[FIX-N]` marker string
before touching a file.

---

## Open Items / Figures Requiring Update

| Item | Detail | Status |
|------|--------|--------|
| `fig_05_model_comparison.png` | Regenerate — underlying numbers are now confirmed final (Table 2b above), figure just needs to be re-rendered from them. | Numbers final; figure regen pending |
| `fig_09_gsshap_importance_fixed.png` | Regenerate to reflect FIX-9's group-level framing in the figure caption/axis labels. | Pending |
| `fig_13_dm_test.png` | Regenerate to show the full raw+FDR table, not just raw-significant pairs. | Pending |
| **DM table discrepancy (new, v5.1)** | Appendix W6 (`readme/README_v4_full.md`) reports a *second* DM-with-correction table for the same model pairs, with different p-values and added Cohen's d (e.g. LSTM vs Mamba: p=0.0199, p_bh=0.0497, p_bonf=0.1989, d=−0.4197 — vs. Figure 13's p_raw=0.0024, p_FDR=0.0251 for the same pair). **Do not merge these tables.** Needs a code-level check of whether W6 uses a different test statistic, bootstrap resampling, or data subset than the FIX-10 implementation in `05_prediction.py`, before either table is trusted as canonical. | **Open — unresolved** |
| FSD seed coverage | `sadaf/augmentation/copula.py` and `sadaf/augmentation/mbb.py` do not yet take an explicit `seed` parameter (only `train_vae()`/`vae_augment()` do, via FIX-21). This is the likely reason FSD still isn't reproducible *within* a single call site across separate runs. | Recommended follow-up |
| Regularisation grid | Confirmed best config: `dropout=0.2, weight_decay=0.0001` → RMSE=1.4073 (GRU-based grid, Appendix W3). Not yet reflected in any figure/table above. | Add to Appendix if regularisation is reported in the manuscript |

Run `python scripts/10_figures.py` after resolving the above to regenerate all figures.

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
- BusinessKorea (Apr 2026). *Naver's Search Market Share Hits 64% While Google Ranked 2nd with 29% Share* (InternetTrend data), used for the RQ0 market-context framing — verify before submission.
