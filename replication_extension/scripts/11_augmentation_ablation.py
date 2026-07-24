"""
11_augmentation_ablation.py — Section 8.2 follow-up: does the
augmentation pipeline actually improve out-of-sample forecasting,
or does it only satisfy the Frechet Sequence Distance quality gate?

The manuscript's regression-stage comparison (Table 7, Section 6.5)
trains every architecture on the ~870-sequence augmented corpus.
Section 8.2 flags, as an explicit limitation, that the paper "does
not report a direct comparison between architectures trained on the
augmented corpus and the same architectures trained on the 174 real
sequences alone." This script runs that missing control condition.

Design
------
- The group-aware train/val/test split (group_time_split) is built
  ONCE and reused for both conditions and every seed, so REAL and
  AUGMENTED are compared on the IDENTICAL held-out test set
  (n = 24 sequences, as in the paper) and the identical real
  validation set.
- Two training regimes per architecture, per seed:
    REAL      — trained on the 174 real training sequences only.
                val_loader = real validation set (early stopping is
                therefore already "real"-driven, matching the
                original trainer's default behaviour when
                real_val_loader is not supplied).
    AUGMENTED — trained on the ~870-sequence augmented corpus
                (identical augment_pipeline call and hyperparameters
                as 05_prediction.py / 09_robustness.py), with early
                stopping driven by the real validation set
                (real_val_loader) exactly as in the paper.
- Because REAL and AUGMENTED share the same seed and the same test
  set, per-architecture, per-seed forecasts are paired: this lets us
  run a Diebold-Mariano test (paired-loss-differential) for
  AUGMENTED vs REAL per architecture per seed, IN ADDITION TO a
  Wilcoxon signed-rank test on paired RMSE across the seed panel per
  architecture (does augmentation reliably move the loss up or down,
  independent of exact effect size).
- Reports:
    (a) per-seed / per-architecture / per-condition RMSE (raw table)
    (b) mean +/- SD RMSE per architecture per condition, and their
        difference (AUGMENTED - REAL; negative = augmentation helps)
    (c) Diebold-Mariano AUGMENTED-vs-REAL, one test per architecture
        per seed, plus Benjamini-Hochberg correction across all
        (architecture x seed) comparisons
    (d) Wilcoxon signed-rank test on paired per-seed RMSE
        (AUGMENTED vs REAL), per architecture, across the seed panel

Usage:
    python scripts/11_augmentation_ablation.py \
        --data_path "data/3월성과데이터(샘플).xlsx" \
        --seeds 42 1 7 123 2024

Runtime note: like 10_multiseed_stability.py, this trains 5 neural
architectures x 2 conditions (real / augmented) x N_SEEDS (default
5) = 50 model trainings, plus 2 linear/MLP baselines x 2 conditions
x N_SEEDS = 20 more. Budget accordingly.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
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
from sadaf.training.trainer import train_model, eval_reg, diebold_mariano

warnings.filterwarnings("ignore")

FEATURES = ["CTR", "CVR", "Depth", "log_cost",
            "log_impression", "hour_sin", "hour_cos"]
D_IN = len(FEATURES)
DEFAULT_SEEDS = [42, 1, 7, 123, 2024]


def build_fixed_split(df_roas: pd.DataFrame):
    """Build the ONE group-aware split shared across both conditions
    and every seed, so REAL vs AUGMENTED is never confounded with a
    different partition of the data."""
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
          f"test:{len(Xte)}  (shared across both conditions, all seeds)")
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)


def _fit_model_registry(seed, tr_l, va_l, real_va_l, te_l):
    """Train the five neural architectures under a given (loader,
    early-stopping) configuration and return {name: (metrics, preds,
    targets)}."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_registry = {
        "BayesianLSTM": BayesianLSTM(D_IN, dropout=0.4),
        "LSTM":         LSTMForecaster(D_IN),
        "GRU":          GRUForecaster(D_IN),
        "BiLSTM":       LSTMForecaster(D_IN, bidirectional=True),
        "Mamba":        MambaForecaster(D_IN, d_model=64, n_layers=3, d_state=8),
    }
    out = {}
    for name, model in model_registry.items():
        torch.manual_seed(seed)
        np.random.seed(seed)
        trained, _ = train_model(
            model, tr_l, va_l, epochs=120, patience=12,
            real_val_loader=real_va_l, verbose=False)
        m, p, t = eval_reg(trained, te_l)
        out[name] = (m, p, t)
        print(f"    seed={seed}  {name:<14} RMSE={m['RMSE']:.4f}")
    return out


def run_real_only(seed: int, split):
    """Condition REAL: train directly on the 174 real training
    sequences. val_loader IS the real validation set, so early
    stopping already matches the paper's real-data-driven stopping
    rule (real_val_loader left as None reproduces that default)."""
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = split
    torch.manual_seed(seed)
    np.random.seed(seed)

    sc = MinMaxScaler()
    N, T, D = Xtr.shape
    Xtr_n = sc.fit_transform(Xtr.reshape(-1, D)).reshape(Xtr.shape)
    Xva_n = np.clip(sc.transform(Xva.reshape(-1, D)).reshape(Xva.shape), 0, 1)
    Xte_n = np.clip(sc.transform(Xte.reshape(-1, D)).reshape(Xte.shape), 0, 1)

    bs = min(32, len(Xtr_n))
    tr_l = DataLoader(SeqDataset(Xtr_n, Ytr), batch_size=bs, shuffle=True)
    va_l = DataLoader(SeqDataset(Xva_n, Yva), batch_size=bs)
    te_l = DataLoader(SeqDataset(Xte_n, Yte), batch_size=bs)

    reg_out = _fit_model_registry(seed, tr_l, va_l, real_va_l=None, te_l=te_l)

    # linear / MLP baselines — deterministic given random_state=seed
    Xtr_f = Xtr_n.reshape(len(Xtr_n), -1)
    Xte_f = Xte_n.reshape(len(Xte_n), -1)
    for name, clf in [
        ("Ridge", Ridge(alpha=1.0)),
        ("MLP",   MLPRegressor(hidden_layer_sizes=(128, 64),
                                max_iter=300, random_state=seed)),
    ]:
        clf.fit(Xtr_f, Ytr)
        p_pred = clf.predict(Xte_f)
        m = {"RMSE": float(np.sqrt(mean_squared_error(Yte, p_pred))),
             "MAE":  float(np.mean(np.abs(p_pred - Yte))),
             "R2":   float(r2_score(Yte, p_pred))}
        reg_out[name] = (m, p_pred, Yte)
        print(f"    seed={seed}  {name:<14} RMSE={m['RMSE']:.4f}")

    return reg_out


def run_augmented(seed: int, split, target_n: int):
    """Condition AUGMENTED: reproduces the paper's pipeline exactly
    — reference GRU retrained per seed, augment_pipeline regenerates
    the corpus, early stopping driven by the real validation set."""
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = split
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    val_size = min(len(X_aug_n) // 10, len(X_aug_n))
    aug_va_l = DataLoader(SeqDataset(X_aug_n[:val_size], Y_aug[:val_size]),
                           batch_size=bs)

    reg_out = _fit_model_registry(seed, tr_l, aug_va_l, real_va_l=real_va_l, te_l=te_l)

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
        reg_out[name] = (m, p_pred, Yte)
        print(f"    seed={seed}  {name:<14} RMSE={m['RMSE']:.4f}")

    return reg_out


def run_one_seed(seed: int, split, target_n: int):
    print(f"\n  -- REAL-ONLY (n=174) --")
    real_out = run_real_only(seed, split)
    print(f"\n  -- AUGMENTED (n~{target_n}) --")
    aug_out = run_augmented(seed, split, target_n)

    rows = []
    dm_rows = []
    for name in real_out:
        m_real, p_real, t_real = real_out[name]
        m_aug, p_aug, t_aug = aug_out[name]
        rows.append({"seed": seed, "architecture": name, "condition": "real",
                     **m_real})
        rows.append({"seed": seed, "architecture": name, "condition": "augmented",
                     **m_aug})

        # paired DM test: augmented vs real, on the identical test targets
        # (t_real == t_aug by construction, since both draw from the same
        # fixed test split)
        n_min = min(len(t_real), len(t_aug))
        e_real = t_real[:n_min] - p_real[:n_min]
        e_aug = t_aug[:n_min] - p_aug[:n_min]
        dm_v, pv = diebold_mariano(e_aug, e_real)
        dm_rows.append({"seed": seed, "architecture": name,
                         "dm_augmented_vs_real": dm_v, "p": pv,
                         "rmse_diff_aug_minus_real": m_aug["RMSE"] - m_real["RMSE"]})

    return rows, dm_rows


def summarize(result_df: pd.DataFrame, dm_df: pd.DataFrame):
    print("\n══ Ablation summary: REAL vs AUGMENTED ═════════════")
    summary = (result_df.groupby(["architecture", "condition"])["RMSE"]
               .agg(mean_RMSE="mean", sd_RMSE="std", n_seeds="count")
               .reset_index())
    pivot = summary.pivot(index="architecture", columns="condition",
                           values="mean_RMSE")
    if "real" in pivot.columns and "augmented" in pivot.columns:
        pivot["diff_aug_minus_real"] = pivot["augmented"] - pivot["real"]
        pivot["pct_change"] = 100 * pivot["diff_aug_minus_real"] / pivot["real"]
    print(pivot.round(4).to_string())
    print("\n  Negative diff_aug_minus_real  → augmentation IMPROVES RMSE")
    print("  Positive diff_aug_minus_real  → augmentation HURTS RMSE "
          "(consistent with memorizing synthetic patterns)")

    # DM correction across all (architecture x seed) comparisons
    _, p_bh, _, _ = multipletests(dm_df["p"].values, method="fdr_bh")
    dm_df = dm_df.copy()
    dm_df["p_bh"] = p_bh
    print("\n  Per-seed Diebold-Mariano (augmented vs real), FDR-corrected:")
    print(dm_df.sort_values(["architecture", "seed"])
          [["architecture", "seed", "dm_augmented_vs_real", "p", "p_bh",
            "rmse_diff_aug_minus_real"]]
          .round(4).to_string(index=False))

    # Wilcoxon signed-rank on paired per-seed RMSE, per architecture
    print("\n  Wilcoxon signed-rank (paired RMSE, augmented vs real) "
          "per architecture across the seed panel:")
    wilcoxon_rows = []
    for arch in result_df["architecture"].unique():
        sub = result_df[result_df["architecture"] == arch]
        real_vals = sub[sub["condition"] == "real"].sort_values("seed")["RMSE"].values
        aug_vals = sub[sub["condition"] == "augmented"].sort_values("seed")["RMSE"].values
        if len(real_vals) != len(aug_vals) or len(real_vals) < 2:
            continue
        try:
            stat, pval = wilcoxon(aug_vals, real_vals)
        except ValueError as e:
            stat, pval = np.nan, np.nan
            print(f"    {arch}: Wilcoxon could not be computed ({e})")
            continue
        wilcoxon_rows.append({"architecture": arch, "wilcoxon_stat": stat,
                               "p": pval,
                               "median_diff": float(np.median(aug_vals - real_vals))})
        print(f"    {arch:<14} W={stat:.3f}  p={pval:.4f}  "
              f"median(aug-real)={np.median(aug_vals - real_vals):+.4f}")

    return pivot, dm_df, pd.DataFrame(wilcoxon_rows)


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: augmentation ablation (real-only vs augmented corpus)")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--target_n", type=int, default=800)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--out_csv", type=str,
                         default="table_ablation_raw.csv")
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)
    split = build_fixed_split(df_roas)

    all_rows, all_dm = [], []
    for seed in args.seeds:
        print(f"\n══ Seed {seed} ══════════════════════════════════")
        rows, dm_rows = run_one_seed(seed, split, args.target_n)
        all_rows.extend(rows)
        all_dm.extend(dm_rows)

    result_df = pd.DataFrame(all_rows)
    dm_df = pd.DataFrame(all_dm)
    result_df.to_csv(args.out_csv, index=False)
    dm_df.to_csv(args.out_csv.replace(".csv", "_dm.csv"), index=False)
    print(f"\nSaved per-seed raw results → {args.out_csv}")
    print(f"Saved DM comparisons → {args.out_csv.replace('.csv', '_dm.csv')}")

    pivot, dm_df, wilcoxon_df = summarize(result_df, dm_df)
    pivot.to_csv(args.out_csv.replace(".csv", "_summary.csv"))
    wilcoxon_df.to_csv(args.out_csv.replace(".csv", "_wilcoxon.csv"), index=False)
    print(f"Saved summary → {args.out_csv.replace('.csv', '_summary.csv')}")
    print(f"Saved Wilcoxon tests → {args.out_csv.replace('.csv', '_wilcoxon.csv')}")

    print("\n✅ Augmentation ablation (Section 8.2 follow-up) complete.")


if __name__ == "__main__":
    main()
