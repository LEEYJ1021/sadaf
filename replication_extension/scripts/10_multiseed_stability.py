"""
10_multiseed_stability.py — Section 8.2 follow-up: multi-seed
stability of the Table 7 / H4b architecture ranking.

The manuscript's regression-stage comparison (Table 7, Section 6.5)
comes from a single train/val/test partition and a single random
seed (42). Section 8.2 flags this explicitly as an unresolved
limitation: "this study does not report the ranking's stability
across repeated random seeds." This script closes that gap.

Design
------
- The group-aware train/val/test split (group_time_split) is built
  ONCE and reused for every seed, so every seed is compared on the
  IDENTICAL held-out test set (n = 24 sequences, as in the paper).
  Only the augmentation draw, model initialization, dropout, and
  minibatch shuffling vary across seeds — the quantities the
  reviewer's comment is actually about.
- For each seed: retrain the reference GRU used by augment_pipeline,
  regenerate the augmented training corpus (augmentation itself is
  stochastic and, per Section 8.4, not currently seed-controllable
  inside the pipeline — that lack of control is itself part of what
  this script is checking the consequences of), then train all five
  neural architectures from Table 7 plus the two linear/MLP
  baselines, and record test RMSE/MAE/R2 for each.
- Reports, across the seed panel:
    (a) per-seed / per-architecture RMSE (raw table, for the
        appendix)
    (b) mean +/- SD RMSE per architecture
    (c) how often each architecture ranks #1 (lowest RMSE)
    (d) Kendall's W (rank concordance across seeds) as a single
        stability statistic — W close to 1 means the ranking is
        essentially the same across seeds, W close to 0 means the
        ranking is close to random from seed to seed
    (e) a Friedman test (nonparametric repeated-measures ANOVA
        analogue) on whether architecture RMSEs differ significantly
        across the seed panel

Usage:
    python scripts/10_multiseed_stability.py \
        --data_path "data/3월성과데이터(샘플).xlsx" \
        --seeds 42 1 7 123 2024

Runtime note: this trains 7 models x N_SEEDS times (default 5), each
with its own augmentation regeneration. Expect this to take
substantially longer than a single run of 05_prediction.py --
budget accordingly (this is the "several-hour" analysis mentioned
in the accompanying discussion).
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import friedmanchisquare
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import (build_sequences, time_split,
                                  group_time_split, SeqDataset)
from sadaf.augmentation.pipeline import augment_pipeline
from sadaf.models.gru import GRUForecaster
from sadaf.models.lstm import LSTMForecaster, BayesianLSTM
from sadaf.models.mamba import MambaForecaster
from sadaf.training.trainer import train_model, eval_reg

warnings.filterwarnings("ignore")

FEATURES = ["CTR", "CVR", "Depth", "log_cost",
            "log_impression", "hour_sin", "hour_cos"]
D_IN = len(FEATURES)
DEFAULT_SEEDS = [42, 1, 7, 123, 2024]


def build_fixed_split(df_roas: pd.DataFrame, target_n: int):
    """Build the ONE group-aware split shared across all seeds, so
    every seed is evaluated on the same held-out test sequences."""
    X_reg, Y_reg, gids_reg = build_sequences(
        df_roas, "log_ROAS", seq_len=4, features=FEATURES)

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = group_time_split(
        X_reg, Y_reg, gids_reg)
    if len(Xva) < 10 or len(Xte) < 10:
        print("  ⚠ group_time_split produced a small fold — "
              "falling back to index split (60/80 fractions)")
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(
            X_reg, Y_reg, 0.60, 0.80)

    print(f"  Fixed split — train:{len(Xtr)}  val:{len(Xva)}  "
          f"test:{len(Xte)}  (shared across all seeds)")
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)


def run_one_seed(seed: int, split, target_n: int):
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = split
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── reference GRU for augmentation, retrained for this seed ──
    sc0 = MinMaxScaler()
    N0, T0, D0 = Xtr.shape
    Xtr_n0 = sc0.fit_transform(Xtr.reshape(-1, D0)).reshape(Xtr.shape)
    Xva_n0 = np.clip(sc0.transform(Xva.reshape(-1, D0)).reshape(Xva.shape), 0, 1)
    bs0 = min(32, len(Xtr_n0))
    ref_gru = GRUForecaster(D_IN)
    ref_gru, _ = train_model(
        ref_gru,
        DataLoader(SeqDataset(Xtr_n0, Ytr), batch_size=bs0, shuffle=True),
        DataLoader(SeqDataset(Xva_n0, Yva), batch_size=bs0),
        epochs=50, patience=8, verbose=False)

    # ── regenerate the augmented corpus for this seed ──
    X_aug, Y_aug = augment_pipeline(
        Xtr.astype(np.float32), Ytr.astype(np.float32),
        target_n=max(target_n, len(Xtr) * 5), ref_model=ref_gru)

    sc = MinMaxScaler()
    N_a, T_a, D_a = X_aug.shape
    X_aug_n = sc.fit_transform(X_aug.reshape(-1, D_a)).reshape(X_aug.shape)
    X_va_n = np.clip(sc.transform(Xva.reshape(-1, D_a)).reshape(Xva.shape), 0, 1)
    X_te_n = np.clip(sc.transform(Xte.reshape(-1, D_a)).reshape(Xte.shape), 0, 1)

    bs = min(128, len(X_aug_n))
    tr_l = DataLoader(SeqDataset(X_aug_n, Y_aug), batch_size=bs, shuffle=True)
    real_va_l = DataLoader(SeqDataset(X_va_n, Yva), batch_size=bs)
    te_l = DataLoader(SeqDataset(X_te_n, Yte), batch_size=bs)

    # keep the same 5% synthetic-val slice convention as 05_prediction.py
    val_size = min(len(X_aug_n) // 10, len(X_aug_n))
    aug_va_l = DataLoader(SeqDataset(X_aug_n[:val_size], Y_aug[:val_size]),
                           batch_size=bs)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model_registry = {
        "BayesianLSTM": BayesianLSTM(D_IN, dropout=0.4),
        "LSTM":         LSTMForecaster(D_IN),
        "GRU":          GRUForecaster(D_IN),
        "BiLSTM":       LSTMForecaster(D_IN, bidirectional=True),
        "Mamba":        MambaForecaster(D_IN, d_model=64, n_layers=3, d_state=8),
    }

    rows = []
    for name, model in model_registry.items():
        torch.manual_seed(seed)
        np.random.seed(seed)
        trained, _ = train_model(
            model, tr_l, aug_va_l, epochs=120, patience=12,
            real_val_loader=real_va_l, verbose=False)
        m, _, _ = eval_reg(trained, te_l)
        rows.append({"seed": seed, "architecture": name, **m})
        print(f"    seed={seed}  {name:<14} RMSE={m['RMSE']:.4f}")

    # linear / MLP baselines (sklearn) — deterministic given random_state=seed
    Xtr_f = X_aug_n.reshape(len(X_aug_n), -1)
    Xte_f = X_te_n.reshape(len(X_te_n), -1)
    for name, clf in [
        ("Ridge", Ridge(alpha=1.0)),
        ("MLP",   MLPRegressor(hidden_layer_sizes=(128, 64),
                                max_iter=300, random_state=seed)),
    ]:
        clf.fit(Xtr_f, Y_aug)
        p_pred = clf.predict(Xte_f)
        m = {"RMSE": float(np.sqrt(mean_squared_error(Yte, p_pred))),
             "MAE":  float(np.mean(np.abs(p_pred - Yte))),
             "R2":   float(r2_score(Yte, p_pred))}
        rows.append({"seed": seed, "architecture": name, **m})
        print(f"    seed={seed}  {name:<14} RMSE={m['RMSE']:.4f}")

    return rows


def summarize(result_df: pd.DataFrame):
    print("\n══ Cross-seed summary ══════════════════════════════")
    summary = (result_df.groupby("architecture")["RMSE"]
               .agg(mean_RMSE="mean", sd_RMSE="std", n_seeds="count")
               .reset_index()
               .sort_values("mean_RMSE"))
    print(summary.round(4).to_string(index=False))

    # how often each architecture ranks #1 (lowest RMSE) within a seed
    ranks = (result_df.pivot(index="seed", columns="architecture", values="RMSE")
             .rank(axis=1, method="min"))
    win_pct = (ranks == 1).mean(axis=0).sort_values(ascending=False) * 100
    print("\n  % of seeds where architecture ranks #1 (lowest RMSE):")
    print(win_pct.round(1).to_string())

    # Kendall's W (coefficient of concordance) across seed rankings
    m, n = ranks.shape  # m = seeds, n = architectures
    rank_sums = ranks.sum(axis=0).values
    S = np.sum((rank_sums - rank_sums.mean()) ** 2)
    W = 12 * S / (m ** 2 * (n ** 3 - n)) if n > 1 else np.nan
    print(f"\n  Kendall's W (rank concordance across {m} seeds, "
          f"{n} architectures) = {W:.4f}")
    print("    W→1: ranking essentially identical across seeds "
          "(Table 7 ordering is stable).")
    print("    W→0: ranking close to random from seed to seed "
          "(Table 7 ordering is largely a single-seed artifact).")

    # Friedman test: do architecture RMSEs differ across the seed panel?
    wide = result_df.pivot(index="seed", columns="architecture", values="RMSE")
    try:
        stat, pval = friedmanchisquare(*[wide[c].values for c in wide.columns])
        print(f"\n  Friedman test across architectures (blocks = seeds): "
              f"chi2={stat:.4f}, p={pval:.4f}")
        print("    A significant result (p<.05) says architecture RMSEs are "
              "not interchangeable across the seed panel — consistent with "
              "(though not proof of) a genuine, seed-independent ranking.")
    except ValueError as e:
        print(f"\n  Friedman test could not be computed ({e}). "
              "This typically means too few seeds were run — use at "
              "least 3, ideally 5+.")

    return summary, win_pct, W


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: multi-seed stability of Table 7 (H4b)")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--target_n", type=int, default=800)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--out_csv", type=str,
                         default="table7_multiseed_raw.csv")
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)
    split = build_fixed_split(df_roas, args.target_n)

    all_rows = []
    for seed in args.seeds:
        print(f"\n══ Seed {seed} ══════════════════════════════════")
        all_rows.extend(run_one_seed(seed, split, args.target_n))

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved per-seed raw results → {args.out_csv}")

    summary, win_pct, W = summarize(result_df)
    summary.to_csv(args.out_csv.replace(".csv", "_summary.csv"), index=False)
    print(f"Saved summary → {args.out_csv.replace('.csv', '_summary.csv')}")

    print("\n✅ Multi-seed stability check (Section 8.2 follow-up) complete.")


if __name__ == "__main__":
    main()
