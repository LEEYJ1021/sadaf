"""
09_robustness.py — Weakness-supplement robustness checks:
LOGO-CV (W1c), regularisation grid (W3), DM multiple-comparison
correction with Cohen's d (W6).

Usage:
    python scripts/09_robustness.py --data_path data/3월성과데이터(샘플).xlsx
"""
import argparse
import copy
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader

from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import build_sequences, time_split
from sadaf.augmentation.pipeline import augment_pipeline
from sadaf.models.gru import GRUForecaster
from sadaf.models.lstm import LSTMForecaster, BayesianLSTM
from sadaf.models.mamba import MambaForecaster
from sadaf.training.trainer import train_model, eval_reg, diebold_mariano
from sadaf.data.sequence import SeqDataset  # [FIX-16] moved from trainer.py
from sadaf.config import DEVICE  # [FIX-20] DEVICE lives in config.py, not imported before

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FEATURES = ["CTR", "CVR", "Depth", "log_cost",
            "log_impression", "hour_sin", "hour_cos"]
D_IN = len(FEATURES)


def run_logo_cv(df_roas: pd.DataFrame, seq_len: int = 4):
    print("\n══ W1c: Leave-One-Ad-Group-Out CV ══════════════════")
    hagg = df_roas[df_roas["Hours"] != 0].groupby(
        ["ad_group_id", "Hours"]).agg(
        impression=("impression", "sum"), click=("click", "sum"),
        cost=("cost", "sum"), CTR=("CTR", "mean"), CVR=("CVR", "mean"),
        ROAS=("ROAS", "mean"), Depth=("Depth", "mean")).reset_index()
    hagg["log_ROAS"] = np.log1p(hagg["ROAS"])
    hagg["log_cost"] = np.log1p(hagg["cost"])
    hagg["log_impression"] = np.log1p(hagg["impression"])
    hagg["hour_sin"] = np.sin(2 * np.pi * hagg["Hours"] / 24)
    hagg["hour_cos"] = np.cos(2 * np.pi * hagg["Hours"] / 24)

    grp_seqs = {}
    for grp in hagg["ad_group_id"].unique():
        g = hagg[hagg["ad_group_id"] == grp].sort_values("Hours")
        if len(g) < seq_len + 2:
            continue
        X_g = g[FEATURES].fillna(0).values.astype(np.float32)
        Y_g = g["log_ROAS"].fillna(0).values.astype(np.float32)
        grp_seqs[grp] = [(X_g[i:i + seq_len], Y_g[i + seq_len])
                          for i in range(len(g) - seq_len)]

    logo_rmses = []
    for held_grp in grp_seqs:
        train_seqs = [s for g, ss in grp_seqs.items() if g != held_grp for s in ss]
        test_seqs = grp_seqs[held_grp]
        if len(train_seqs) < 10 or len(test_seqs) < 1:
            continue
        Xtr = np.stack([s[0] for s in train_seqs])
        Ytr = np.array([s[1] for s in train_seqs])
        Xte = np.stack([s[0] for s in test_seqs])
        Yte = np.array([s[1] for s in test_seqs])
        sc = MinMaxScaler()
        N_l, T_l, D_l = Xtr.shape
        Xtr_n = sc.fit_transform(Xtr.reshape(-1, D_l)).reshape(Xtr.shape)
        Xte_n = np.clip(sc.transform(Xte.reshape(-1, D_l)).reshape(Xte.shape), 0, 1)
        bs = min(32, len(Xtr_n))
        ld_tr = DataLoader(SeqDataset(Xtr_n, Ytr), batch_size=bs, shuffle=True)
        ld_te = DataLoader(SeqDataset(Xte_n, Yte), batch_size=bs)
        m = GRUForecaster(D_IN)
        m, _ = train_model(m, ld_tr, ld_tr, epochs=50, patience=7)
        met, _, _ = eval_reg(m, ld_te)
        logo_rmses.append({"group": held_grp, "RMSE": met["RMSE"], "n_test": len(Yte)})

    logo_df = pd.DataFrame(logo_rmses)
    print(f"  LOGO-CV RMSE = {logo_df['RMSE'].mean():.4f} ± {logo_df['RMSE'].std():.4f}  "
          f"(n_groups = {len(logo_df)})")
    return logo_df


def run_regularisation_grid(X_tr_n, Y_tr, X_va_n, Y_va, X_te_n, Y_te,
                             tr_loader, va_loader, te_loader):
    print("\n══ W3: Regularisation Grid (dropout × weight_decay) ══")
    dropout_vals = [0.2, 0.3, 0.4]
    wd_vals = [1e-4, 5e-4, 1e-3]
    reg_grid = {}
    for do in dropout_vals:
        for wd in wd_vals:
            key = f"do={do}_wd={wd}"
            m = GRUForecaster(D_IN, dropout=do).to(DEVICE)  # [FIX-19]
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
            criterion = nn.HuberLoss()
            best_v, best_state, pat = float("inf"), None, 0
            for ep in range(60):  # [FIX-19b] move batches to DEVICE explicitly
                m.train()
                for Xb, Yb in tr_loader:
                    Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(m(Xb), Yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                m.eval()
                vl = []
                with torch.no_grad():
                    for Xb, Yb in va_loader:
                        Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                        vl.append(criterion(m(Xb), Yb).item())
                vm = np.mean(vl)
                if vm < best_v:
                    best_v, best_state, pat = vm, copy.deepcopy(m.state_dict()), 0
                else:
                    pat += 1
                if pat >= 8:
                    break
            m.load_state_dict(best_state)
            met, _, _ = eval_reg(m, te_loader)
            reg_grid[key] = {**met, "dropout": do, "wd": wd}

    grid_df = pd.DataFrame(reg_grid).T
    best_key = grid_df["RMSE"].idxmin()
    print(f"  Best regularisation: {best_key}  RMSE={grid_df.loc[best_key, 'RMSE']:.4f}")
    return grid_df


def run_dm_correction(reg_preds: dict):
    print("\n══ W6: DM Multiple-Comparison Correction ═══════════")
    pairs = {}
    for m1, m2 in combinations(reg_preds.keys(), 2):
        p1, t1 = reg_preds[m1]
        p2, t2 = reg_preds[m2]
        n_min = min(len(t1), len(t2))
        e1, e2 = t1[:n_min] - p1[:n_min], t2[:n_min] - p2[:n_min]
        dm_v, pv = diebold_mariano(e1, e2)
        d_sq = e1 ** 2 - e2 ** 2
        cohen_d = d_sq.mean() / (d_sq.std(ddof=1) + 1e-8)
        pairs[f"{m1}_vs_{m2}"] = {"dm": dm_v, "p": pv, "cohen_d": cohen_d}

    pairs_df = pd.DataFrame(pairs).T
    pvals = pairs_df["p"].values.astype(float)
    _, p_bonf, _, _ = multipletests(pvals, method="bonferroni")
    _, p_bh, _, _ = multipletests(pvals, method="fdr_bh")
    pairs_df["p_bonf"] = p_bonf
    pairs_df["p_bh"] = p_bh

    print(pairs_df.sort_values("p")[["p", "p_bh", "p_bonf", "cohen_d"]]
          .round(4).to_string())
    n_sig_raw = (pairs_df["p"] < 0.05).sum()
    n_sig_bh = (pairs_df["p_bh"] < 0.05).sum()
    n_sig_bonf = (pairs_df["p_bonf"] < 0.05).sum()
    print(f"\n  Significant pairs — raw: {n_sig_raw}  BH-FDR: {n_sig_bh}  "
          f"Bonferroni: {n_sig_bonf}  (out of {len(pairs_df)})")
    return pairs_df


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: robustness checks (LOGO-CV, reg grid, DM correction)")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--target_n", type=int, default=800)
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)

    run_logo_cv(df_roas)

    # [FIX-17] features must be passed as keyword; return is a 3-tuple
    X_reg, Y_reg, _reg_gids = build_sequences(
        df_roas, "log_ROAS", seq_len=4, features=FEATURES)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(X_reg, Y_reg)
    if len(Xva) < 10 or len(Xte) < 10:
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(X_reg, Y_reg, 0.60, 0.80)

    ref_gru = GRUForecaster(D_IN)
    sc0 = MinMaxScaler()
    N0, T0, D0 = Xtr.shape
    Xtr_n0 = sc0.fit_transform(Xtr.reshape(-1, D0)).reshape(Xtr.shape)
    Xva_n0 = np.clip(sc0.transform(Xva.reshape(-1, D0)).reshape(Xva.shape), 0, 1)
    bs0 = min(32, len(Xtr_n0))
    ref_gru, _ = train_model(
        ref_gru,
        DataLoader(SeqDataset(Xtr_n0, Ytr), batch_size=bs0, shuffle=True),
        DataLoader(SeqDataset(Xva_n0, Yva), batch_size=bs0),
        epochs=50, patience=8)

    # [FIX-18] augment_pipeline()'s actual parameter is ref_model, not ref_lstm
    X_aug, Y_aug = augment_pipeline(
        Xtr.astype(np.float32), Ytr.astype(np.float32),
        target_n=max(args.target_n, len(Xtr) * 5), ref_model=ref_gru)

    sc = MinMaxScaler()
    N_a, T_a, D_a = X_aug.shape
    X_aug_n = sc.fit_transform(X_aug.reshape(-1, D_a)).reshape(X_aug.shape)
    X_va_n = np.clip(sc.transform(Xva.reshape(-1, D_a)).reshape(Xva.shape), 0, 1)
    X_te_n = np.clip(sc.transform(Xte.reshape(-1, D_a)).reshape(Xte.shape), 0, 1)

    bs = min(128, len(X_aug_n))
    tr_l = DataLoader(SeqDataset(X_aug_n, Y_aug), batch_size=bs, shuffle=True)
    va_l = DataLoader(SeqDataset(X_va_n, Yva), batch_size=bs)
    te_l = DataLoader(SeqDataset(X_te_n, Yte), batch_size=bs)

    run_regularisation_grid(X_aug_n, Y_aug, X_va_n, Yva, X_te_n, Yte, tr_l, va_l, te_l)

    model_registry = {
        "BayesianLSTM": BayesianLSTM(D_IN, dropout=0.4),
        "LSTM": LSTMForecaster(D_IN),
        "GRU": GRUForecaster(D_IN),
        "BiLSTM": LSTMForecaster(D_IN, bidirectional=True),
        "Mamba": MambaForecaster(D_IN, d_model=64, n_layers=3, d_state=8),
    }
    reg_preds = {}
    for name, model in model_registry.items():
        trained, _ = train_model(model, tr_l, va_l, epochs=120, patience=12)
        _, p, t = eval_reg(trained, te_l)
        reg_preds[name] = (p, t)

    run_dm_correction(reg_preds)
    print("\n✅ Robustness checks complete.")


if __name__ == "__main__":
    main()