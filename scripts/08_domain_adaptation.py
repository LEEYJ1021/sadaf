"""
08_domain_adaptation.py — H6: cross-campaign domain shift (Search vs.
Shopping) via KS test, plus frozen-encoder fine-tuning transfer.

Usage:
    python scripts/08_domain_adaptation.py --data_path data/3월성과데이터(샘플).xlsx
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import build_sequences, time_split
from sadaf.models.gru import GRUForecaster
from sadaf.training.trainer import train_model, eval_reg
from sadaf.data.sequence import SeqDataset  # [FIX-14] moved from trainer.py

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURES = ["CTR", "CVR", "Depth", "log_cost",
            "log_impression", "hour_sin", "hour_cos"]
D_IN = len(FEATURES)


def run_ks_test(df: pd.DataFrame):
    print("\n══ H6: Domain Shift Analysis ═══════════════════════")
    df_shop = df[df["campaign_type_label"] == "Shopping"]
    df_search = df[df["campaign_type_label"] == "Search"]
    # [FIX-15] features must be passed as keyword (positional 3rd slot is
    # seq_len in the current signature); return is a 3-tuple (X, Y, group_ids)
    X_shop, Y_shop, _shop_gids = build_sequences(
        df_shop, "log_ROAS", seq_len=4, features=FEATURES)
    X_src, Y_src, _src_gids = build_sequences(
        df_search, "log_ROAS", seq_len=4, features=FEATURES)
    print(f"  Shopping: {X_shop.shape}  Search: {X_src.shape}")

    n_sig = 0
    for i, feat in enumerate(FEATURES):
        ks, p = ks_2samp(X_shop[:, -1, i], X_src[:, -1, i])
        sig = "*" if p < 0.05 else "ns"
        if p < 0.05:
            n_sig += 1
        print(f"  {feat:<18} KS={ks:.4f} p={p:.4e} {sig}")

    h6_supported = n_sig >= 6
    print(f"\n  H6: {'SUPPORTED ✓' if h6_supported else 'PARTIAL ⚬'}  "
          f"({n_sig}/{len(FEATURES)} features p<0.05)")
    return X_shop, Y_shop, X_src, Y_src


def run_transfer(X_src, Y_src, X_tgt, Y_tgt):
    print("\n══ Domain Adaptation: Search → Shopping ════════════")
    if len(X_src) <= 10 or len(X_tgt) <= 10:
        print("  Insufficient sequences for transfer experiment.")
        return

    sc = MinMaxScaler()
    N_s, T_s, D_s = X_src.shape
    X_src_n = sc.fit_transform(X_src.reshape(-1, D_s)).reshape(X_src.shape)
    X_tgt_n = np.clip(sc.transform(X_tgt.reshape(-1, D_s)).reshape(X_tgt.shape), 0, 1)

    (Xt_tr, Yt_tr), (Xt_va, Yt_va), (Xt_te, Yt_te) = time_split(
        X_tgt_n, Y_tgt, 0.60 if len(X_tgt_n) < 50 else 0.70,
        0.80 if len(X_tgt_n) < 50 else 0.85)

    bs_src = min(32, len(X_src_n))
    ld_src_tr = DataLoader(SeqDataset(X_src_n, Y_src), batch_size=bs_src, shuffle=True)
    ld_src_va = DataLoader(SeqDataset(Xt_va, Yt_va), batch_size=bs_src)

    src_model = GRUForecaster(D_IN)
    print("  Step 1: Training source model on Search...")
    src_model, _ = train_model(src_model, ld_src_tr, ld_src_va, epochs=80, patience=10)

    ld_tgt_te = DataLoader(SeqDataset(Xt_te, Yt_te), batch_size=bs_src)
    m_naive, _, _ = eval_reg(src_model, ld_tgt_te)
    print(f"  Naive transfer RMSE = {m_naive['RMSE']:.4f}")

    all_params = list(src_model.named_parameters())
    n_freeze = len(all_params) // 2
    for i, (_, param) in enumerate(all_params):
        param.requires_grad = (i >= n_freeze)

    ld_tgt_tr = DataLoader(SeqDataset(Xt_tr, Yt_tr), batch_size=bs_src, shuffle=True)
    print("  Step 2: Fine-tuning on Shopping (50% frozen)...")
    src_model, _ = train_model(src_model, ld_tgt_tr, ld_src_va,
                                epochs=50, patience=8, lr=5e-5)
    m_adapted, _, _ = eval_reg(src_model, ld_tgt_te)
    rmse_gain = (m_naive["RMSE"] - m_adapted["RMSE"]) / m_naive["RMSE"] * 100
    print(f"  Adapted transfer RMSE = {m_adapted['RMSE']:.4f}  "
          f"(gain: {rmse_gain:+.1f}% vs naive transfer)")
    print("  NOTE: modest improvement; primary contribution is theoretical")
    print("  justification for domain-adaptive design, not a performance claim.")


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: domain shift + adaptation (H6)")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)
    X_shop, Y_shop, X_src, Y_src = run_ks_test(df)
    run_transfer(X_src, Y_src, X_shop, Y_shop)
    print("\n✅ Domain shift + adaptation (H6) complete.")


if __name__ == "__main__":
    main()