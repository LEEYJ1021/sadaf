# SADAF — v4/v5 Full Analysis (Korea / March-2025 Case Study)


_Generated 2026-07-07 12:35 from `run_sadaf_v5_korea_case_study.py`. Patches applied: FIX-9 (H5 group-level verdict), FIX-10 (FDR-corrected DM), FIX-22/23 (05_prediction.py: explicit seed refixation before model instantiation and before each model's training loop, for H4b winner reproducibility), FIX-11/12/13 (pre-existing psm/mediation/moderation API mismatches in 03_causal.py), FIX-14/15 (08_domain_adaptation.py import + build_sequences() API mismatches), FIX-16/17/18/19/19b/20 (09_robustness.py import, build_sequences(), augment_pipeline() keyword, and CPU/GPU device mismatches), FIX-21 (sadaf/augmentation/pipeline.py: explicit seed fixation in train_vae()/vae_augment() for FSD reproducibility)._



> **Why Korea, why March 2025, why a single (Naver-based) advertiser?**
> This is presented as the intentional scope of a *boundary-condition case
> study*, not an incidental limitation. According to InternetTrend data
> reported by BusinessKorea (Apr 2026), Naver held an average **63.8%**
> share of Korean search volume in March 2025 versus Google's **28.7%**
> (Feb 2025: Naver 65.1%) — a platform-concentration structure with no
> equivalent in the Google-dominated markets (>90% Google share) that
> most cold-start / computational-advertising literature is built on.
> The dataset's own ad-group spend concentration (HHI = 201.0 on a
> 0-10,000 scale) and campaign-type mix (Search 19.4%
> / Shopping 78.8% / Zero-cost
> 1.8%) are reported here as descriptive
> scope indicators, not as a representativeness claim about the Korean
> market as a whole.
>
> Source (verify before submission): BusinessKorea, "Naver's Search Market
> Share Hits 64% While Google Ranked 2nd with 29% Share" (Apr 16, 2026),
> citing InternetTrend monthly tracking data.



## Research Questions — v4 (Korea / March-2025 Case-Study Framing)

**RQ0 (framing, not a tested hypothesis).** Do causal, predictive, and
explainability patterns established primarily in Google-dominated
advertising markets replicate under a structurally different,
single-platform-concentrated search ecosystem? March 2025 Korea (Naver
≈63.8% share) is treated as a natural boundary-condition test case.

**RQ1–RQ3 (unchanged methods, reframed claim scope).** H1 (CTR→conversion,
PSM+IPW), H2 (Depth mediation), H3 (campaign-type moderation) are reported
as causal structure *conditional on* a platform-concentrated market, not as
population-level claims about advertising in general.

**RQ4 + RQ4d (domain-gap as first-class evidence, not a nuisance).**
H4a–H4c unchanged. **H4d:** Does the augmentation-to-real domain gap
itself differ systematically across architectures in a way diagnostic of
overfitting risk under extreme cold-start sparsity (N_train=174 real
sequences)? Reported explicitly, not minimized.

**RQ5 (unchanged, group-level).** H5 tests 2 independent HSIC-group-level
distributions, not 7 per-feature distributions (FIX-9).

**RQ6 (unchanged).** Search vs. Shopping domain shift (KS test).

**RQ7 (explicit external-validity boundary).** LOGO-CV (W1c) and synthetic
multi-advertiser (W2b) support generalization *within* this single-platform,
single-month case; claims beyond it are explicitly out of scope.


## Appendix A — EDA (§1)

```
Loading data from: /home/yjlee/Research/Advertise_codeNdata/3월성과데이터(샘플).xlsx
── Dataset summary ──────────────────────────────────────
  Total rows    :   89,675
  Paid rows     :   32,494
  ROAS > 0      :    9,071  (27.9% of paid)
  Conversion %  : 11.77%
  Zero-ROAS %   : 72.1%
─────────────────────────────────────────────────────────

=== 데이터 크기 ===
행: 89,675 / 열: 32

=== 컬럼 목록 ===
  1. Date (datetime64[us])
  2. Hours (int64)
  3. customer_id (int64)
  4. campaign_id (str)
  5. ad_group_id (str)
  6. ad_id (str)
  7. impression (int64)
  8. click (int64)
  9. cost (int64)
  10. sum_of_ad_rank (int64)
  11. conversion_count (int64)
  12. sales_by_conversion (int64)
  13. CTR (float64)
  14. CVR (float64)
  15. ROAS (float64)
  16. Depth (float64)
  17. CPC (float64)
  18. CPA (float64)
  19. has_conversion (int64)
  20. log_impression (float64)
  21. log_click (float64)
  22. log_cost (float64)
  23. log_CTR (float64)
  24. log_CVR (float64)
  25. log_ROAS (float64)
  26. log_CPC (float64)
  27. log_conversion_count (float64)
  28. hour_sin (float64)
  29. hour_cos (float64)
  30. campaign_type (str)
  31. campaign_type_label (str)
  32. hour_bin (category)

=== 결측값 ===
결측값 없음

=== 기술통계 ===
                       count       mean         std  ...     75%         max      cv
impression           89675.0    648.034    4886.034  ...  169.00    187026.0   7.540
click                89675.0      3.598      32.917  ...    1.00      1951.0   9.150
cost                 89675.0   3230.415   18910.979  ...  620.00    865200.0   5.854
CTR                  89675.0      1.783       6.291  ...    0.97       100.0   3.529
CVR                  89675.0      3.857      24.403  ...    0.00      4600.0   6.326
ROAS                 89675.0    671.213    9303.012  ...    0.00   1339000.0  13.860
Depth                89675.0      6.028       7.344  ...    6.90       137.0   1.218
conversion_count     89675.0      0.364       2.875  ...    0.00       179.0   7.890
sales_by_conversion  89675.0  28938.367  374064.113  ...    0.00  26397240.0  12.926

[9 rows x 9 columns]

── Dataset summary ──────────────────────────────
  Total rows    :   89,675
  Paid rows     :   32,494
  ROAS > 0      :    9,071  (27.9% of paid)
  Conversion %  : 11.77%
  ROAS=0 (spare): 72.1%  ← cold-start context, not a flaw

=== 캠페인 유형 분포 ===
campaign_type_label
Shopping     70693
Search       17373
Zero-cost     1609
Name: count, dtype: int64

=== 캠페인별 성과 요약 (상위 10, by cost) ===
                campaign_id  impression  click     cost      CTR       CVR        ROAS  conversion  ad_count
cmp-a001-02-000000006247200     2297020  13138 40690970 2.605636 21.723545 1579.012523        2457        38
cmp-a001-02-000000008278062      698027   4648 37305690 4.252160 22.765953  353.259475         990        32
cmp-a001-02-000000006489917    16564798  23514 33752820 1.251594 10.366900 1441.825584        3091       180
cmp-a001-02-000000008697378       93237   1704 10945970 5.211853 31.970594 2148.396152         603        40
cmp-a001-02-000000006516851     1439396   6828 10623420 1.489286 10.679955 2059.559107         664        14
cmp-a001-02-000000001589513     5567843   5166  9424110 1.658041 12.336551 1794.893663         513        62
cmp-a001-02-000000006247191     1084070   2221  9090000 5.072549 13.549499  422.794466         318        30
cmp-a001-01-000000005308661      190274   6935  7846650 4.292489 11.412017 3111.859227         866        24
cmp-a001-02-000000008025659      622024   9503  7168510 2.176230 20.540471 1502.042984        2575        65
cmp-a001-02-000000006679269     1011346   3451  6239160 0.398229  4.607500  491.385000         147         6

✅ EDA complete.
```

## Appendix B — ZINB Structure Diagnosis (§2)

```
Loading data from: /home/yjlee/Research/Advertise_codeNdata/3월성과데이터(샘플).xlsx
── Dataset summary ──────────────────────────────────────
  Total rows    :   89,675
  Paid rows     :   32,494
  ROAS > 0      :    9,071  (27.9% of paid)
  Conversion %  : 11.77%
  Zero-ROAS %   : 72.1%
─────────────────────────────────────────────────────────

=== Structural Zero-Inflation Diagnostics ===
  Zero-ROAS rate          : 72.1%
  ROAS mean               : 1852.37
  ROAS variance           : 236661259.29
  Overdispersion (var/mean): 127761.02

  ZINB converged via lbfgs (SE valid ✓)  AIC=71958.2
  ZINB AIC=71958.2  BIC=72025.3
  ΔAIC(ZIP−ZINB)=-2798.9  (>10 = ZINB strongly preferred)

 ====================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------
inflate_const        5.6169      0.079     71.543      0.000       5.463       5.771
inflate_log_CTR     -0.1895      0.018    -10.708      0.000      -0.224      -0.155
inflate_log_cost    -0.5814      0.009    -65.692      0.000      -0.599      -0.564
const                1.6407      0.133     12.378      0.000       1.381       1.901
log_CTR              0.4725      0.036     12.996      0.000       0.401       0.544
log_cost            -0.2157      0.004    -51.699      0.000      -0.224      -0.207
log_impression       0.2184      0.016     13.315      0.000       0.186       0.251
alpha                0.0119      0.015      0.797      0.425      -0.017       0.041
====================================================================================

✅ ZINB diagnosis complete.
```

## §3 — Causal Results: H1 (PSM+IPW), H2 (Mediation), H3 (Moderation) [FIX-11/12/13]

```
── Dataset summary ──────────────────────────────────────
  Total rows    :   89,675
  Paid rows     :   32,494
  ROAS > 0      :    9,071  (27.9% of paid)
  Conversion %  : 11.77%
  Zero-ROAS %   : 72.1%
─────────────────────────────────────────────────────────

══ H1: PSM + Doubly Robust IPW [FIX-11] ═══════════
  PSM-ATT = 0.1347  95% Boot CI = [0.1254, 0.1434]
  n_matched = 14987  H1 (PSM, corroborating): SUPPORTED ✓
  IPW-ATT = 0.1286  [PRIMARY ESTIMATOR — doubly robust]
  DR consistency (|IPW-ATT − PSM-ATT| < 0.05): ✓ consistent

  Table S1: Covariate Balance (PSM + DR correction)
     Covariate  SMD_before  SMD_after  p_before  p_after                                               Balance
log_impression     -1.5459    -0.4299       0.0      0.0 ✗ substantial — DR corrected (primary estimator: IPW)
      log_cost     -0.2372    -0.5752       0.0      0.0 ✗ substantial — DR corrected (primary estimator: IPW)
         Depth     -0.1636     0.1253       0.0      0.0                             ⚠ residual — DR corrected
ctype_Shopping     -0.4350     0.0801       0.0      0.0                                            ✓ balanced

══ H2: Mediation Analysis [FIX-12] ══════════════════
  a = -0.3077  b = -0.0861  indirect (a×b) = 0.0265  CI = [0.0200, 0.0337]
  Type: Negative suppressor (a<0, b<0, a×b>0)
  Interpretation: High-CTR ads reduce browsing depth (a<0): immediate-click campaigns bypass deliberate browsing. Among ads generating depth, deeper browsing reduces conversion (b<0), indicating depth proxies decision hesitancy, not engagement. The positive indirect product constitutes a negative suppressor.
  Proportion mediated: -42.8% (appendix only; sign direction is the primary finding)
  H2: SUPPORTED ✓ (bootstrap CI excludes 0)

══ H3: Moderation Analysis [FIX-13] ═════════════════
  β_interaction = 0.3860  p = 0.0000  H3: SUPPORTED ✓
  ME_Shopping = 0.563  ME_Search = 0.949
  R² = 0.6551  n = 9069

✅ Causal analyses (H1–H3) complete.
```

## §4 — Two-Stage Prediction: H4a–H4c + Domain-Gap (H4d) [FIX-10]

```
── Dataset summary ──────────────────────────────────────
  Total rows    :   89,675
  Paid rows     :   32,494
  ROAS > 0      :    9,071  (27.9% of paid)
  Conversion %  : 11.77%
  Zero-ROAS %   : 72.1%
─────────────────────────────────────────────────────────

══ H4a: Classification Stage ═══════════════════════
    Ep  10: train=0.4759  val_aug=0.5855
    Ep  20: train=0.4407  val_aug=0.5758
    Early stop @ epoch 25  (best aug val=0.5675)
    Ep  10: train=0.4772  val_aug=0.5760
    Ep  20: train=0.4598  val_aug=0.5803
    Early stop @ epoch 26  (best aug val=0.5646)

  Table 2a: Classification Results
                     AUC      F1      AP  Thresh
LR-Cls            0.6143  0.3653  0.3016    0.12
LSTM-Cls          0.6115  0.3026  0.3062    0.24
BayesianLSTM-Cls  0.5894  0.3151  0.2723    0.28
MLP-Cls           0.5445  0.2951  0.2989    0.50

  H4a: NULL (boundary) ⚬  (LSTM-Cls AUC=0.6115 vs LR-Cls AUC=0.6143)

══ H4b/H4c: Regression Stage ═══════════════════════
  REG SEQ_LEN=4: (222, 4, 7)
  REG SEQ_LEN=6: (125, 6, 7)
  Split sizes — train:174  val:24  test:24
    Ep  10: train=0.8854  val_aug=0.7678
    Ep  20: train=0.6535  val_aug=0.6106
    Ep  30: train=0.5789  val_aug=0.5896
    Early stop @ epoch 35  (best aug val=0.5620)
  Augmentation: 174 → ~870 (+232 per method)
  [1/3] Training β-VAE ...
    VAE Ep 100: loss=53648.92
    VAE Ep 200: loss=45476.11
    VAE Ep 300: loss=38695.28
  [2/3] Gaussian Copula ...
    Copula KS validation: mean_KS=0.097 (lower = more realistic)
  [3/3] Moving Block Bootstrap ...
  FSD = -0.0465  [PASS ✓]  (threshold: <2.0 accept, >5.0 reject)
    Ep  10: train=1.3299  val_aug=0.9399  val_real=1.2951
    Ep  20: train=1.1522  val_aug=0.8418  val_real=0.8354
    Ep  30: train=1.1686  val_aug=0.7789  val_real=0.9955
    Ep  40: train=1.1475  val_aug=0.8221  val_real=0.7878
    Ep  50: train=1.0925  val_aug=0.7720  val_real=0.7739
    Early stop @ epoch 54  (best real val=0.7525)
  BayesianLSTM   RMSE=1.3420  MAE=1.1063  R²=0.6729
    Ep  10: train=0.9620  val_aug=0.9548  val_real=0.9425
    Ep  20: train=0.8182  val_aug=0.7491  val_real=0.8606
    Ep  30: train=0.8360  val_aug=0.7191  val_real=0.7070
    Early stop @ epoch 34  (best real val=0.7047)
  LSTM           RMSE=1.2099  MAE=0.9608  R²=0.7342
    Ep  10: train=1.0351  val_aug=0.8493  val_real=1.1376
    Ep  20: train=0.8405  val_aug=0.7853  val_real=0.7387
    Ep  30: train=0.8887  val_aug=0.7230  val_real=0.8117
    Ep  40: train=0.8623  val_aug=0.7483  val_real=0.7547
    Early stop @ epoch 48  (best real val=0.6873)
  GRU            RMSE=1.3984  MAE=1.1450  R²=0.6449
    Ep  10: train=0.8709  val_aug=0.8318  val_real=0.9570
    Ep  20: train=0.7762  val_aug=0.8029  val_real=0.7921
    Ep  30: train=0.8039  val_aug=0.7214  val_real=0.8078
    Early stop @ epoch 35  (best real val=0.7587)
  BiLSTM         RMSE=1.4998  MAE=1.1629  R²=0.5915
    Ep  10: train=0.9713  val_aug=0.9390  val_real=1.1628
    Ep  20: train=0.8121  val_aug=0.7252  val_real=0.8968
    Ep  30: train=0.7160  val_aug=0.6404  val_real=0.8487
    Ep  40: train=0.6601  val_aug=0.5378  val_real=0.8862
    Early stop @ epoch 47  (best real val=0.7910)
  Mamba          RMSE=1.6356  MAE=1.3308  R²=0.5142
  Ridge          RMSE=1.6033  R²=0.5331
  MLP            RMSE=1.7086  R²=0.4699

  Table 2b: Regression Results (group-split, SL=4)
                RMSE     MAE      R2
LSTM          1.2099  0.9608  0.7342
BayesianLSTM  1.3420  1.1063  0.6729
GRU           1.3984  1.1450  0.6449
BiLSTM        1.4998  1.1629  0.5915
Ridge         1.6033  1.2684  0.5331
Mamba         1.6356  1.3308  0.5142
MLP           1.7086  1.3538  0.4699

  ── Domain-gap report (train–val gap) ──────────────
              best_epoch_real  best_val_real  best_val_aug  final_train  gap_real  gap_aug
model                                                                                     
BayesianLSTM               42         0.7525        0.7441       1.0936   -0.2821  -0.3494
LSTM                       22         0.7047        0.7133       0.8080   -0.0862  -0.0947
GRU                        36         0.6873        0.7131       0.8312   -0.1591  -0.1181
BiLSTM                     23         0.7587        0.6964       0.8477   -0.0802  -0.1513
Mamba                      35         0.7910        0.4564       0.6138    0.1142  -0.1574

  H4b: SUPPORTED ✓  (LSTM RMSE=1.2099 vs Ridge RMSE=1.6033)

  ── DM Comparisons: raw p + BH-FDR correction [FIX-10] ──
  BayesianLSTM vs LSTM          DM= 3.1782  p_raw=0.0042*  p_FDR=0.0293*  → LSTM
  BayesianLSTM vs GRU           DM=-1.1323  p_raw=0.2692   p_FDR=0.3140   → BayesianLSTM
  BayesianLSTM vs BiLSTM        DM=-1.6517  p_raw=0.1122   p_FDR=0.1472   → BayesianLSTM
  BayesianLSTM vs Mamba         DM=-2.3879  p_raw=0.0255*  p_FDR=0.0537   → BayesianLSTM
  BayesianLSTM vs Ridge         DM=-2.1653  p_raw=0.0410*  p_FDR=0.0615   → BayesianLSTM
  BayesianLSTM vs MLP           DM=-2.9221  p_raw=0.0077*  p_FDR=0.0316*  → BayesianLSTM
  LSTM         vs GRU           DM=-2.7851  p_raw=0.0105*  p_FDR=0.0316*  → LSTM
  LSTM         vs BiLSTM        DM=-2.7881  p_raw=0.0105*  p_FDR=0.0316*  → LSTM
  LSTM         vs Mamba         DM=-3.4113  p_raw=0.0024*  p_FDR=0.0251*  → LSTM
  LSTM         vs Ridge         DM=-2.9124  p_raw=0.0078*  p_FDR=0.0316*  → LSTM
  LSTM         vs MLP           DM=-3.4587  p_raw=0.0021*  p_FDR=0.0251*  → LSTM
  GRU          vs BiLSTM        DM=-1.5906  p_raw=0.1254   p_FDR=0.1549   → GRU
  GRU          vs Mamba         DM=-2.3405  p_raw=0.0283*  p_FDR=0.0540   → GRU
  GRU          vs Ridge         DM=-2.4195  p_raw=0.0239*  p_FDR=0.0537   → GRU
  GRU          vs MLP           DM=-2.7088  p_raw=0.0125*  p_FDR=0.0329*  → GRU
  BiLSTM       vs Mamba         DM=-2.2002  p_raw=0.0381*  p_FDR=0.0615   → BiLSTM
  BiLSTM       vs Ridge         DM=-2.2157  p_raw=0.0369*  p_FDR=0.0615   → BiLSTM
  BiLSTM       vs MLP           DM=-2.0115  p_raw=0.0561   p_FDR=0.0786   → BiLSTM
  Mamba        vs Ridge         DM= 0.4228  p_raw=0.6763   p_FDR=0.6763   → Ridge
  Mamba        vs MLP           DM=-0.5872  p_raw=0.5628   p_FDR=0.5909   → Mamba
  Ridge        vs MLP           DM=-0.9403  p_raw=0.3568   p_FDR=0.3944   → Ridge

  Summary: 14/21 pairs significant at raw p<0.05; 8/21 remain significant after BH-FDR (test sequences n=24). Report both numbers in the manuscript — do not report raw-only.

✅ Two-stage prediction (H4a–H4c) [FIXED v2] complete.
```

## §5 — Multi-Method Attribution: H5 (group-level) [FIX-9]

```
── Dataset summary ──────────────────────────────────────
  Total rows    :   89,675
  Paid rows     :   32,494
  ROAS > 0      :    9,071  (27.9% of paid)
  Conversion %  : 11.77%
  Zero-ROAS %   : 72.1%
─────────────────────────────────────────────────────────
  Split sizes — train:174  val:24  test:24
  [DIAGNOSTIC] augment_pipeline call site: 07_explainability.py main(), target_n=870, len(Xtr)=174, pid=3619919
  Augmentation: 174 → ~870 (+232 per method)
  [1/3] Training β-VAE ...
    VAE Ep 100: loss=53648.92
    VAE Ep 200: loss=45476.11
    VAE Ep 300: loss=38695.28
  [2/3] Gaussian Copula ...
    Copula KS validation: mean_KS=0.097 (lower = more realistic)
  [3/3] Moving Block Bootstrap ...
  [DIAGNOSTIC] augment_pipeline returned: len(X_aug)=870 (expected ~870)
  [FIX-7] Loading saved model from /home/yjlee/sadaf/figures/best_bayesian_lstm.pt

══ H5: Multi-Method Attribution Comparison [FIXED v3] ══
  Cluster 0 (C0 High-Volume): n=7
  Cluster 1 (C1 High-Conversion): n=9
  Cluster 2 (C2 Click-Rich): n=8

  [1/4] GS-SHAP (HSIC grouping + Shapley) [FIX-3/4a/4b] ...
[GS-SHAP] Computing HSIC feature groups from training data...
  [HSIC] eigengap → K=2 groups (D=7 features)
  Groups: [[np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)], [np.int64(5), np.int64(6)]]  (0.11s)
  [Segmentation] seg_len=1, time_segments=4, n_players=8 (K=2 groups × 4 time segments)
  [Reporting] HSIC groups → raw features: {0: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)], 1: [np.int64(5), np.int64(6)]}
  [Reporting] 2 independent group-level Gini values will be reported; per-feature values inside a group are identical by construction (see gsshap.py FIX-5).

  ── Temporal Gini by cluster (group-level, non-duplicated) ──
  C0 High-Volume: group0[np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]=0.328  group1[np.int64(5), np.int64(6)]=0.317
  C1 High-Conversion: group0[np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]=0.280  group1[np.int64(5), np.int64(6)]=0.342
  C2 Click-Rich: group0[np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]=0.373  group1[np.int64(5), np.int64(6)]=0.290

  ── Temporal Gini by cluster (feature-level, for reference; values repeat within an HSIC group) ──
  C0 High-Volume: CTR=0.328±0.144  CVR=0.328±0.144  Depth=0.328±0.144  log_cost=0.328±0.144  log_impression=0.328±0.144  hour_sin=0.317±0.076  hour_cos=0.317±0.076
  C1 High-Conversion: CTR=0.280±0.100  CVR=0.280±0.100  Depth=0.280±0.100  log_cost=0.280±0.100  log_impression=0.280±0.100  hour_sin=0.342±0.138  hour_cos=0.342±0.138
  C2 Click-Rich: CTR=0.373±0.092  CVR=0.373±0.092  Depth=0.373±0.092  log_cost=0.373±0.092  log_impression=0.373±0.092  hour_sin=0.290±0.088  hour_cos=0.290±0.088

  [2/4] Integrated Gradients ...

  [3/4] Permutation SHAP ...

  [4/4] Attention-based Attribution ...

  ── Method Agreement: Spearman Rank Correlation ────
  C0 High-Volume          avg Spearman ρ = 0.825  (High, 3/3 pairs usable, n=7 ⚠ underpowered)
  C1 High-Conversion      avg Spearman ρ = 0.559  (Moderate, 3/3 pairs usable, n=9 ⚠ underpowered)
  C2 Click-Rich           avg Spearman ρ = 0.813  (High, 3/3 pairs usable, n=8 ⚠ underpowered)

  ⚠ NOTE: ['C0 High-Volume', 'C1 High-Conversion', 'C2 Click-Rich'] have n < 10 test samples. Agreement and Kruskal-Wallis statistics for these clusters should be reported with this caveat, not as unconditional null/positive findings.

  ── Kruskal-Wallis (GS-SHAP, GROUP-LEVEL, primary) [FIX-9] ──
  group0 [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)] p=4.2791e-02 *   (applies identically to all 5 raw features in this group)
  group1 [np.int64(5), np.int64(6)]   p=6.8800e-01 ns   (applies identically to all 2 raw features in this group)

  H5 [FIX-9]: SUPPORTED ✓  (1/2 HSIC GROUP-LEVEL tests significant — NOT a per-feature count; see gsshap.py FIX-5)
  → Caveat: result is influenced by underpowered cluster(s) ['C0 High-Volume', 'C1 High-Conversion', 'C2 Click-Rich']; see note above.
  → Figure 9 (fixed) saved to /home/yjlee/sadaf/figures/fig_09_gsshap_importance_fixed.png

✅ Multi-method attribution (H5) [FIXED v3] complete.
```

## §6 — Cross-Campaign Domain Shift: H6

```
── Dataset summary ──────────────────────────────────────
  Total rows    :   89,675
  Paid rows     :   32,494
  ROAS > 0      :    9,071  (27.9% of paid)
  Conversion %  : 11.77%
  Zero-ROAS %   : 72.1%
─────────────────────────────────────────────────────────

══ H6: Domain Shift Analysis ═══════════════════════
  Shopping: (4247, 4, 7)  Search: (3476, 4, 7)
  CTR                KS=0.2200 p=1.5116e-81 *
  CVR                KS=0.0383 p=7.1042e-03 *
  Depth              KS=0.3749 p=1.1274e-239 *
  log_cost           KS=0.2271 p=6.4111e-87 *
  log_impression     KS=0.1310 p=4.9394e-29 *
  hour_sin           KS=0.0940 p=3.7180e-15 *
  hour_cos           KS=0.0296 p=6.7920e-02 ns

  H6: SUPPORTED ✓  (6/7 features p<0.05)

══ Domain Adaptation: Search → Shopping ════════════
  Step 1: Training source model on Search...
    Ep  10: train=0.5994  val_aug=0.2720
    Early stop @ epoch 17  (best aug val=0.2718)
  Naive transfer RMSE = 1.3176
  Step 2: Fine-tuning on Shopping (50% frozen)...
    Ep  10: train=0.4919  val_aug=0.2724
    Early stop @ epoch 14  (best aug val=0.2720)
  Adapted transfer RMSE = 1.3128  (gain: +0.4% vs naive transfer)
  NOTE: modest improvement; primary contribution is theoretical
  justification for domain-adaptive design, not a performance claim.

✅ Domain shift + adaptation (H6) complete.
```

## Appendix W — Robustness Supplement (LOGO-CV, Regularisation Grid, DM+FDR, Multi-Advertiser)

```
── Dataset summary ──────────────────────────────────────
  Total rows    :   89,675
  Paid rows     :   32,494
  ROAS > 0      :    9,071  (27.9% of paid)
  Conversion %  : 11.77%
  Zero-ROAS %   : 72.1%
─────────────────────────────────────────────────────────

══ W1c: Leave-One-Ad-Group-Out CV ══════════════════
    Ep  10: train=0.7881  val_aug=0.5651
    Ep  20: train=0.6957  val_aug=0.4698
    Early stop @ epoch 29  (best aug val=0.4056)
    Ep  10: train=0.6196  val_aug=0.5130
    Ep  20: train=0.6071  val_aug=0.4479
    Early stop @ epoch 29  (best aug val=0.3765)
    Ep  10: train=1.0160  val_aug=0.8781
    Ep  20: train=0.6320  val_aug=0.4485
    Ep  30: train=0.6079  val_aug=0.4294
    Ep  40: train=0.6633  val_aug=0.4324
    Ep  50: train=0.6254  val_aug=0.4045
    Ep  10: train=1.0923  val_aug=1.0370
    Ep  20: train=0.6092  val_aug=0.3988
    Early stop @ epoch 30  (best aug val=0.3914)
    Ep  10: train=0.8111  val_aug=0.7070
    Ep  20: train=0.5937  val_aug=0.4669
    Ep  30: train=0.5983  val_aug=0.4387
    Ep  40: train=0.6377  val_aug=0.4285
    Ep  50: train=0.5630  val_aug=0.4213
    Ep  10: train=0.7521  val_aug=0.5440
    Ep  20: train=0.5979  val_aug=0.4930
    Early stop @ epoch 25  (best aug val=0.3889)
    Ep  10: train=0.8226  val_aug=0.5478
    Ep  20: train=0.6612  val_aug=0.4842
    Ep  30: train=0.7039  val_aug=0.4326
    Ep  40: train=0.6508  val_aug=0.4289
    Early stop @ epoch 45  (best aug val=0.4169)
    Ep  10: train=0.7088  val_aug=0.5844
    Ep  20: train=0.6657  val_aug=0.3926
    Early stop @ epoch 30  (best aug val=0.3738)
    Ep  10: train=0.7541  val_aug=0.7174
    Ep  20: train=0.8533  val_aug=0.5047
    Early stop @ epoch 28  (best aug val=0.3948)
    Ep  10: train=0.6494  val_aug=0.4779
    Ep  20: train=0.8563  val_aug=0.4544
    Ep  30: train=0.5561  val_aug=0.3893
    Early stop @ epoch 35  (best aug val=0.3739)
    Ep  10: train=0.7032  val_aug=0.6257
    Ep  20: train=0.6757  val_aug=0.4953
    Early stop @ epoch 29  (best aug val=0.3930)
    Ep  10: train=0.8082  val_aug=0.7610
    Ep  20: train=0.6952  val_aug=0.4506
    Ep  30: train=0.6888  val_aug=0.7240
    Early stop @ epoch 32  (best aug val=0.3883)
    Ep  10: train=0.8618  val_aug=0.5306
    Ep  20: train=0.6393  val_aug=0.5314
    Early stop @ epoch 25  (best aug val=0.3825)
    Ep  10: train=0.9506  val_aug=0.6582
    Ep  20: train=0.7025  val_aug=0.4421
    Early stop @ epoch 28  (best aug val=0.3796)
    Ep  10: train=0.6826  val_aug=0.4690
    Ep  20: train=0.6827  val_aug=0.3938
    Ep  30: train=0.5852  val_aug=0.5485
    Early stop @ epoch 33  (best aug val=0.3669)
    Ep  10: train=0.8603  val_aug=0.6749
    Ep  20: train=0.6628  val_aug=0.4948
    Ep  30: train=0.6203  val_aug=0.4351
    Ep  40: train=0.6857  val_aug=0.4188
    Early stop @ epoch 46  (best aug val=0.4074)
    Ep  10: train=0.8822  val_aug=0.5867
    Ep  20: train=0.6653  val_aug=0.5158
    Ep  30: train=0.6941  val_aug=0.4426
    Ep  40: train=0.5695  val_aug=0.4270
    Early stop @ epoch 49  (best aug val=0.4211)
    Ep  10: train=0.7859  val_aug=0.5588
    Ep  20: train=0.7129  val_aug=0.4771
    Early stop @ epoch 21  (best aug val=0.4685)
    Ep  10: train=0.7475  val_aug=0.6393
    Ep  20: train=0.6800  val_aug=0.5693
    Ep  30: train=0.6095  val_aug=0.4584
    Ep  40: train=0.5948  val_aug=0.4464
    Ep  50: train=0.5991  val_aug=0.4367
    Ep  10: train=0.7202  val_aug=0.5759
    Ep  20: train=0.5827  val_aug=0.4626
    Ep  30: train=0.5515  val_aug=0.4234
    Early stop @ epoch 38  (best aug val=0.4110)
    Ep  10: train=0.7429  val_aug=0.5388
    Ep  20: train=0.6833  val_aug=0.4448
    Early stop @ epoch 24  (best aug val=0.3995)
    Ep  10: train=0.7020  val_aug=0.6300
    Ep  20: train=0.7121  val_aug=0.4799
    Early stop @ epoch 28  (best aug val=0.4054)
    Ep  10: train=0.6635  val_aug=0.4485
    Ep  20: train=0.6232  val_aug=0.3857
    Early stop @ epoch 29  (best aug val=0.3681)
    Ep  10: train=0.8017  val_aug=0.5385
    Ep  20: train=0.7215  val_aug=0.4099
    Early stop @ epoch 26  (best aug val=0.4052)
    Ep  10: train=0.7856  val_aug=0.5609
    Ep  20: train=0.7062  val_aug=0.5030
    Ep  30: train=0.6756  val_aug=0.4441
    Early stop @ epoch 39  (best aug val=0.3737)
    Ep  10: train=0.8509  val_aug=0.5108
    Ep  20: train=0.7075  val_aug=0.5038
    Ep  30: train=0.6248  val_aug=0.4658
    Ep  40: train=0.7009  val_aug=0.3768
    Early stop @ epoch 44  (best aug val=0.3455)
    Ep  10: train=0.9227  val_aug=0.5888
    Ep  20: train=0.5790  val_aug=0.4404
    Early stop @ epoch 26  (best aug val=0.3754)
    Ep  10: train=0.9488  val_aug=0.5890
    Ep  20: train=0.7488  val_aug=0.4715
    Early stop @ epoch 22  (best aug val=0.4430)
    Ep  10: train=0.7787  val_aug=0.6090
    Ep  20: train=0.6207  val_aug=0.4437
    Early stop @ epoch 25  (best aug val=0.4083)
    Ep  10: train=0.8807  val_aug=0.5901
    Ep  20: train=0.6079  val_aug=0.5129
    Ep  30: train=0.5615  val_aug=0.4441
    Ep  40: train=0.5873  val_aug=0.4134
    Ep  50: train=0.6544  val_aug=0.4089
    Ep  10: train=0.6385  val_aug=0.4773
    Ep  20: train=0.6402  val_aug=0.5853
    Ep  30: train=0.5071  val_aug=0.4794
    Early stop @ epoch 36  (best aug val=0.3740)
    Ep  10: train=0.8748  val_aug=0.7192
    Ep  20: train=0.6739  val_aug=0.4344
    Early stop @ epoch 25  (best aug val=0.4207)
    Ep  10: train=0.7190  val_aug=0.6100
    Ep  20: train=0.5403  val_aug=0.4085
    Ep  30: train=0.6116  val_aug=0.4312
    Early stop @ epoch 35  (best aug val=0.3788)
    Ep  10: train=0.7782  val_aug=0.6207
    Ep  20: train=0.6676  val_aug=0.4570
    Early stop @ epoch 24  (best aug val=0.4071)
    Ep  10: train=0.7274  val_aug=0.5497
    Ep  20: train=0.5954  val_aug=0.3911
    Early stop @ epoch 29  (best aug val=0.3750)
    Ep  10: train=0.7298  val_aug=0.5005
    Ep  20: train=0.6614  val_aug=0.4074
    Ep  30: train=0.6005  val_aug=0.3880
    Ep  40: train=0.6145  val_aug=0.3517
    Early stop @ epoch 47  (best aug val=0.3517)
    Ep  10: train=0.7499  val_aug=0.6610
    Ep  20: train=0.6368  val_aug=0.4285
    Early stop @ epoch 26  (best aug val=0.4225)
  LOGO-CV RMSE = 1.2427 ± 0.6042  (n_groups = 37)
    Ep  10: train=0.8675  val_aug=0.4941
    Ep  20: train=0.5840  val_aug=0.3715
    Ep  30: train=0.4929  val_aug=0.3671
    Early stop @ epoch 33  (best aug val=0.3465)
  Augmentation: 155 → ~800 (+215 per method)
  [1/3] Training β-VAE ...
    VAE Ep 100: loss=53879.20
    VAE Ep 200: loss=45896.85
    VAE Ep 300: loss=41341.34
  [2/3] Gaussian Copula ...
    Copula KS validation: mean_KS=0.106 (lower = more realistic)
  [3/3] Moving Block Bootstrap ...
  FSD = -1.0057  [PASS ✓]  (threshold: <2.0 accept, >5.0 reject)

══ W3: Regularisation Grid (dropout × weight_decay) ══
  Best regularisation: do=0.2_wd=0.0001  RMSE=1.4073
    Ep  10: train=1.3686  val_aug=1.0819
    Ep  20: train=1.2211  val_aug=0.8931
    Ep  30: train=1.2262  val_aug=0.9303
    Ep  40: train=1.1129  val_aug=0.7989
    Ep  50: train=1.1634  val_aug=0.7672
    Ep  60: train=1.1813  val_aug=0.8246
    Early stop @ epoch 63  (best aug val=0.7524)
    Ep  10: train=0.9344  val_aug=0.8757
    Ep  20: train=0.8764  val_aug=0.7861
    Ep  30: train=0.8293  val_aug=0.7898
    Ep  40: train=0.8678  val_aug=0.8084
    Early stop @ epoch 50  (best aug val=0.7133)
    Ep  10: train=0.9727  val_aug=0.8574
    Ep  20: train=0.9180  val_aug=0.8292
    Ep  30: train=0.8873  val_aug=0.7870
    Ep  40: train=0.8741  val_aug=0.7444
    Early stop @ epoch 45  (best aug val=0.7152)
    Ep  10: train=0.8487  val_aug=0.9669
    Ep  20: train=0.8850  val_aug=0.8564
    Ep  30: train=0.9043  val_aug=0.8776
    Ep  40: train=0.8293  val_aug=0.8171
    Early stop @ epoch 44  (best aug val=0.7860)
    Ep  10: train=1.1252  val_aug=1.1480
    Ep  20: train=0.9291  val_aug=0.9783
    Ep  30: train=0.9389  val_aug=1.1445
    Early stop @ epoch 37  (best aug val=0.8718)

══ W6: DM Multiple-Comparison Correction ═══════════
                             p    p_bh  p_bonf  cohen_d
BayesianLSTM_vs_Mamba   0.0010  0.0065  0.0104  -0.6169
GRU_vs_Mamba            0.0014  0.0065  0.0142  -0.5974
BiLSTM_vs_Mamba         0.0019  0.0065  0.0194  -0.5777
LSTM_vs_Mamba           0.0199  0.0497  0.1989  -0.4197
LSTM_vs_GRU             0.0651  0.1302  0.6508   0.3273
BayesianLSTM_vs_LSTM    0.0962  0.1604  0.9623  -0.2937
LSTM_vs_BiLSTM          0.3021  0.4316  1.0000   0.1798
BayesianLSTM_vs_GRU     0.5198  0.6498  1.0000   0.1116
GRU_vs_BiLSTM           0.5955  0.6616  1.0000  -0.0919
BayesianLSTM_vs_BiLSTM  0.9495  0.9495  1.0000   0.0109

  Significant pairs — raw: 4  BH-FDR: 4  Bonferroni: 3  (out of 10)

✅ Robustness checks complete.
```

## Overfitting / Domain-Gap Policy (explicit statement for reviewers)

This README reports the augmentation-to-real domain gap and the post-FDR Diebold-Mariano results in full, alongside the raw/headline numbers, rather than excluding them. Given N_train=174 real sequences (pre-augmentation), some train/val divergence is expected and is treated as diagnostic evidence about architecture suitability under extreme cold-start sparsity (RQ4d), not as a defect to be edited out.