"""
sadaf/config.py
---------------
Central configuration: all hyperparameters, feature lists, and constants
used across the SADAF pipeline.
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# ── Feature columns used for sequence modelling ───────────────────────────────
FEATURES = [
    "CTR",
    "CVR",
    "Depth",
    "log_cost",
    "log_impression",
    "hour_sin",
    "hour_cos",
]
D_IN = len(FEATURES)   # input dimensionality = 7

# ── Sequence construction ────────────────────────────────────────────────────
SEQ_LEN_DEFAULT = 4    # default look-back window (hours)
SEQ_LEN_ALT     = 6    # alternative length for sensitivity analysis (H4c)

# ── Data split fractions ─────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.85   # cumulative; val = 0.70–0.85, test = 0.85–1.00

# ── Augmentation ─────────────────────────────────────────────────────────────
AUG_TARGET_N   = 800    # target training corpus size after augmentation
VAE_EPOCHS     = 300
VAE_LR         = 3e-4
VAE_BATCH      = 32
VAE_LATENT_DIM = 16
VAE_BETA       = 1.0

MBB_BLOCK_SIZE  = 3

FSD_ACCEPT_THRESHOLD = 2.0   # Fréchet Score Distance pass/warn/reject thresholds
FSD_WARN_THRESHOLD   = 5.0

# ── PSM / IPW (H1) ───────────────────────────────────────────────────────────
PSM_CALIPER_SIGMA = 0.1     # caliper = 0.1 × std(logit propensity score)
PSM_N_BOOT        = 2000    # bootstrap resamples for ATT confidence interval
IPW_CLIP_QUANTILE = 0.99    # clip IPW weights at this quantile

# ── Mediation (H2) ───────────────────────────────────────────────────────────
MED_N_BOOT = 2000

# ── Model hyperparameters ─────────────────────────────────────────────────────
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.2

BAYESIAN_LSTM_DROPOUT   = 0.4    # reviewer-fixed value (was 0.3)
BAYESIAN_TEMPERATURE    = 1.5    # temperature scaling for calibration
BAYESIAN_MC_SAMPLES     = 500

MAMBA_D_MODEL  = 64
MAMBA_N_LAYERS = 3
MAMBA_D_STATE  = 8
MAMBA_DROPOUT  = 0.1

PROTO_HIDDEN   = 64
PROTO_PROJ_DIM = 32
PROTO_EPOCHS   = 100

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_EPOCHS   = 120
TRAIN_LR       = 1e-3
TRAIN_WD       = 1e-4
TRAIN_PATIENCE = 12
TRAIN_BATCH    = 512
REG_BATCH      = 32

# ── Explainability ───────────────────────────────────────────────────────────
N_CLUSTERS       = 3
N_EXPLAIN        = 20    # samples per cluster for attribution
PERM_SHAP_N      = 50    # permutation replicates per feature
INTGRAD_N_STEPS  = 50

# ── Diebold-Mariano ──────────────────────────────────────────────────────────
DM_H         = 1       # forecast horizon
DM_N_BOOT    = 1000    # bootstrap resamples for RMSE CI

# ── Domain adaptation ─────────────────────────────────────────────────────────
DA_FREEZE_FRAC = 0.5   # fraction of layers to freeze during fine-tuning
DA_LR          = 5e-5
DA_EPOCHS      = 50
DA_PATIENCE    = 8
