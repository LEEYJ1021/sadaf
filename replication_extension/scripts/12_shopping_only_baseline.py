"""
12_shopping_only_baseline.py — Section 8.3 follow-up: a Shopping-only
trained forecaster, to decompose the domain-adaptation transfer
result reported in Section 6.8 / 08_domain_adaptation.py.

08_domain_adaptation.py reports a naive Search-to-Shopping transfer
RMSE and a fine-tuned ("adapted") transfer RMSE, but has no
Shopping-only baseline: a model trained (and never exposed to
Search) entirely on Shopping campaign sequences. Without it, the
naive-vs-adapted RMSE range cannot be decomposed into (a) the cost
of genuine cross-campaign domain shift, recoverable by having
Shopping-specific training data at all, versus (b) the intrinsic
difficulty of forecasting Shopping-campaign ROAS on its own,
irreducible even with in-domain training data. This script adds
that missing condition.

Design
------
- Reuses run_ks_test()'s sequence construction from
  08_domain_adaptation.py conceptually: Search sequences (X_src,
  Y_src) are the source domain, Shopping sequences (X_shop, Y_shop)
  are the target domain.
- The Shopping sequences are split ONCE (time_split, same fractions
  used inside run_transfer in 08_domain_adaptation.py) into
  Shopping-train / Shopping-val / Shopping-test. This SAME
  Shopping-test set is used to evaluate all three conditions below,
  so the three RMSEs are directly comparable:
    (1) NAIVE      — GRU trained only on Search, evaluated directly
                      on Shopping-test (no exposure to Shopping data
                      at all).
    (2) ADAPTED    — the same Search-trained GRU, with the later
                      half of its encoder + output head fine-tuned
                      on Shopping-train (50% frozen), matching
                      08_domain_adaptation.py's recipe exactly.
    (3) SHOPPING-ONLY — a fresh GRU trained from scratch on
                      Shopping-train alone (no Search exposure ever),
                      evaluated on the identical Shopping-test set.
- Decomposition logic:
    domain_shift_recoverable_gap = naive_RMSE - shopping_only_RMSE
        (how much of the naive transfer's error is attributable to
        being trained on the wrong domain, i.e. is recoverable by
        having Shopping-specific data)
    adaptation_captured_gap = naive_RMSE - adapted_RMSE
        (how much of that recoverable gap the paper's frozen-encoder
        fine-tuning recipe actually captures)
    residual_intrinsic_difficulty = shopping_only_RMSE
        (the floor: how hard Shopping-campaign forecasting is even
        with in-domain training data — this is what "adapted" can
        approach at best, not beat)
  If shopping_only_RMSE << naive_RMSE, most of the naive-transfer
  error is a genuine, recoverable domain-shift cost. If
  shopping_only_RMSE is close to naive_RMSE, Shopping-campaign
  forecasting is simply hard on its own, and domain adaptation has
  limited headroom regardless of technique.
- Pairwise Diebold-Mariano tests (naive vs shopping-only, adapted vs
  shopping-only, naive vs adapted) on the shared Shopping-test set,
  with Benjamini-Hochberg correction across the three comparisons.
- Repeats across a small seed panel (default 3) for stability, since
  a single train/val/test partition of an already-small Shopping
  sequence pool is itself a limited-sample comparison.

Usage:
    python scripts/12_shopping_only_baseline.py \
        --data_path "data/3월성과데이터(샘플).xlsx" \
        --seeds 42 1 7
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import MinMaxScaler

from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import build_sequences, time_split, SeqDataset
from sadaf.models.gru import GRUForecaster
from sadaf.training.trainer import train_model, eval_reg, diebold_mariano

warnings.filterwarnings("ignore")

FEATURES = ["CTR", "CVR", "Depth", "log_cost",
            "log_impression", "hour_sin", "hour_cos"]
D_IN = len(FEATURES)
DEFAULT_SEEDS = [42, 1, 7]


def build_domain_sequences(df: pd.DataFrame):
    """Reproduces 08_domain_adaptation.py's run_ks_test() sequence
    construction (Search = source, Shopping = target), without the
    KS-test reporting, since that diagnostic is unaffected by this
    script's addition."""
    df_shop = df[df["campaign_type_label"] == "Shopping"]
    df_search = df[df["campaign_type_label"] == "Search"]
    X_shop, Y_shop, _shop_gids = build_sequences(
        df_shop, "log_ROAS", seq_len=4, features=FEATURES)
    X_src, Y_src, _src_gids = build_sequences(
        df_search, "log_ROAS", seq_len=4, features=FEATURES)
    print(f"  Shopping: {X_shop.shape}  Search: {X_src.shape}")
    return X_src, Y_src, X_shop, Y_shop


def build_fixed_shopping_split(X_shop, Y_shop):
    """Build the ONE Shopping train/val/test split shared across all
    three conditions and every seed, matching the fractions used in
    08_domain_adaptation.py's run_transfer()."""
    frac1 = 0.60 if len(X_shop) < 50 else 0.70
    frac2 = 0.80 if len(X_shop) < 50 else 0.85
    (Xt_tr, Yt_tr), (Xt_va, Yt_va), (Xt_te, Yt_te) = time_split(
        X_shop, Y_shop, frac1, frac2)
    print(f"  Shopping split — train:{len(Xt_tr)}  val:{len(Xt_va)}  "
          f"test:{len(Xt_te)}  (shared across all three conditions, all seeds)")
    return (Xt_tr, Yt_tr), (Xt_va, Yt_va), (Xt_te, Yt_te)


def run_one_seed(seed: int, X_src, Y_src, shop_split):
    (Xt_tr, Yt_tr), (Xt_va, Yt_va), (Xt_te, Yt_te) = shop_split
    torch.manual_seed(seed)
    np.random.seed(seed)

    # scaler fit on Search (source) to preserve 08_domain_adaptation.py's
    # naive-transfer preprocessing exactly
    sc = MinMaxScaler()
    N_s, T_s, D_s = X_src.shape
    X_src_n = sc.fit_transform(X_src.reshape(-1, D_s)).reshape(X_src.shape)
    Xt_tr_n = np.clip(sc.transform(Xt_tr.reshape(-1, D_s)).reshape(Xt_tr.shape), 0, 1)
    Xt_va_n = np.clip(sc.transform(Xt_va.reshape(-1, D_s)).reshape(Xt_va.shape), 0, 1)
    Xt_te_n = np.clip(sc.transform(Xt_te.reshape(-1, D_s)).reshape(Xt_te.shape), 0, 1)

    bs_src = min(32, len(X_src_n))
    ld_src_tr = DataLoader(SeqDataset(X_src_n, Y_src), batch_size=bs_src, shuffle=True)
    ld_src_va = DataLoader(SeqDataset(Xt_va_n, Yt_va), batch_size=bs_src)
    ld_tgt_te = DataLoader(SeqDataset(Xt_te_n, Yt_te), batch_size=bs_src)

    # ── (1) NAIVE: Search-trained, never sees Shopping ──
    print("    [1/3] Training source model on Search (naive transfer)...")
    src_model = GRUForecaster(D_IN)
    src_model, _ = train_model(src_model, ld_src_tr, ld_src_va,
                                epochs=80, patience=10, verbose=False)
    m_naive, p_naive, t_naive = eval_reg(src_model, ld_tgt_te)
    print(f"      NAIVE RMSE = {m_naive['RMSE']:.4f}")

    # ── (2) ADAPTED: fine-tune 50% of the same Search-trained model ──
    print("    [2/3] Fine-tuning on Shopping (50% frozen)...")
    all_params = list(src_model.named_parameters())
    n_freeze = len(all_params) // 2
    for i, (_, param) in enumerate(all_params):
        param.requires_grad = (i >= n_freeze)
    ld_tgt_tr = DataLoader(SeqDataset(Xt_tr_n, Yt_tr), batch_size=bs_src, shuffle=True)
    src_model, _ = train_model(src_model, ld_tgt_tr, ld_src_va,
                                epochs=50, patience=8, lr=5e-5, verbose=False)
    m_adapted, p_adapted, t_adapted = eval_reg(src_model, ld_tgt_te)
    print(f"      ADAPTED RMSE = {m_adapted['RMSE']:.4f}")

    # ── (3) SHOPPING-ONLY: fresh model, Shopping data only, never Search ──
    print("    [3/3] Training Shopping-only baseline (no Search exposure)...")
    sc_shop = MinMaxScaler()
    Xt_tr_shop_n = sc_shop.fit_transform(
        Xt_tr.reshape(-1, D_s)).reshape(Xt_tr.shape)
    Xt_va_shop_n = np.clip(
        sc_shop.transform(Xt_va.reshape(-1, D_s)).reshape(Xt_va.shape), 0, 1)
    Xt_te_shop_n = np.clip(
        sc_shop.transform(Xt_te.reshape(-1, D_s)).reshape(Xt_te.shape), 0, 1)
    bs_shop = min(32, len(Xt_tr_shop_n))
    ld_shop_tr = DataLoader(SeqDataset(Xt_tr_shop_n, Yt_tr), batch_size=bs_shop, shuffle=True)
    ld_shop_va = DataLoader(SeqDataset(Xt_va_shop_n, Yt_va), batch_size=bs_shop)
    ld_shop_te = DataLoader(SeqDataset(Xt_te_shop_n, Yt_te), batch_size=bs_shop)

    torch.manual_seed(seed)
    np.random.seed(seed)
    shop_model = GRUForecaster(D_IN)
    shop_model, _ = train_model(shop_model, ld_shop_tr, ld_shop_va,
                                 epochs=80, patience=10, verbose=False)
    m_shop, p_shop, t_shop = eval_reg(shop_model, ld_shop_te)
    print(f"      SHOPPING-ONLY RMSE = {m_shop['RMSE']:.4f}")

    row = {
        "seed": seed,
        "naive_RMSE": m_naive["RMSE"], "naive_MAE": m_naive["MAE"], "naive_R2": m_naive["R2"],
        "adapted_RMSE": m_adapted["RMSE"], "adapted_MAE": m_adapted["MAE"], "adapted_R2": m_adapted["R2"],
        "shopping_only_RMSE": m_shop["RMSE"], "shopping_only_MAE": m_shop["MAE"], "shopping_only_R2": m_shop["R2"],
        "domain_shift_recoverable_gap": m_naive["RMSE"] - m_shop["RMSE"],
        "adaptation_captured_gap": m_naive["RMSE"] - m_adapted["RMSE"],
        "adaptation_capture_ratio": (
            (m_naive["RMSE"] - m_adapted["RMSE"]) / (m_naive["RMSE"] - m_shop["RMSE"])
            if abs(m_naive["RMSE"] - m_shop["RMSE"]) > 1e-9 else float("nan")
        ),
    }

    # ── pairwise DM tests on the SAME Shopping-test targets ──
    # note: t_naive, t_adapted are on the sc(Search)-normalized test set,
    # t_shop is on the sc_shop(Shopping)-normalized test set; targets
    # (log_ROAS) are identical in both since only X was rescaled, but we
    # align lengths defensively.
    n_min = min(len(t_naive), len(t_adapted), len(t_shop))
    e_naive = t_naive[:n_min] - p_naive[:n_min]
    e_adapted = t_adapted[:n_min] - p_adapted[:n_min]
    e_shop = t_shop[:n_min] - p_shop[:n_min]

    dm_rows = []
    for label, e1, e2 in [
        ("naive_vs_shopping_only", e_naive, e_shop),
        ("adapted_vs_shopping_only", e_adapted, e_shop),
        ("naive_vs_adapted", e_naive, e_adapted),
    ]:
        dm_v, pv = diebold_mariano(e1, e2)
        dm_rows.append({"seed": seed, "comparison": label, "dm": dm_v, "p": pv})

    return row, dm_rows


def summarize(result_df: pd.DataFrame, dm_df: pd.DataFrame):
    print("\n══ Shopping-only baseline: decomposition summary ═══")
    cols = ["naive_RMSE", "adapted_RMSE", "shopping_only_RMSE",
            "domain_shift_recoverable_gap", "adaptation_captured_gap",
            "adaptation_capture_ratio"]
    summary = result_df[cols].agg(["mean", "std"]).T
    print(summary.round(4).to_string())

    print("\n  Reading guide:")
    print("    domain_shift_recoverable_gap = naive_RMSE - shopping_only_RMSE")
    print("      → total error attributable to being trained on the wrong")
    print("        domain at all (large positive = genuine domain shift cost).")
    print("    adaptation_captured_gap = naive_RMSE - adapted_RMSE")
    print("      → how much of that gap the paper's fine-tuning recipe")
    print("        actually recovers.")
    print("    adaptation_capture_ratio = adaptation_captured_gap / "
          "domain_shift_recoverable_gap")
    print("      → fraction of the recoverable domain-shift gap captured")
    print("        by adaptation; near 1.0 = adaptation recovers nearly all")
    print("        of it; near 0 = adaptation recovers almost none of it;")
    print("        shopping_only_RMSE remains the floor neither naive nor")
    print("        adapted can beat without in-domain Shopping training data.")

    _, p_bh, _, _ = multipletests(dm_df["p"].values, method="fdr_bh")
    dm_df = dm_df.copy()
    dm_df["p_bh"] = p_bh
    print("\n  Pairwise Diebold-Mariano tests (FDR-corrected across all "
          "seed x comparison pairs):")
    print(dm_df.sort_values(["comparison", "seed"])
          [["comparison", "seed", "dm", "p", "p_bh"]]
          .round(4).to_string(index=False))

    return summary, dm_df


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: Shopping-only baseline (domain-shift decomposition)")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--out_csv", type=str,
                         default="table_shopping_only_raw.csv")
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)
    X_src, Y_src, X_shop, Y_shop = build_domain_sequences(df)
    shop_split = build_fixed_shopping_split(X_shop, Y_shop)

    if len(X_src) <= 10 or len(shop_split[0][0]) < 10:
        print("  Insufficient sequences for the domain-shift decomposition "
              "experiment (need > 10 Search sequences and a non-trivial "
              "Shopping-train split). Aborting.")
        return

    all_rows, all_dm = [], []
    for seed in args.seeds:
        print(f"\n══ Seed {seed} ══════════════════════════════════")
        row, dm_rows = run_one_seed(seed, X_src, Y_src, shop_split)
        all_rows.append(row)
        all_dm.extend(dm_rows)

    result_df = pd.DataFrame(all_rows)
    dm_df = pd.DataFrame(all_dm)
    result_df.to_csv(args.out_csv, index=False)
    dm_df.to_csv(args.out_csv.replace(".csv", "_dm.csv"), index=False)
    print(f"\nSaved per-seed raw results → {args.out_csv}")
    print(f"Saved DM comparisons → {args.out_csv.replace('.csv', '_dm.csv')}")

    summary, dm_df = summarize(result_df, dm_df)
    summary.to_csv(args.out_csv.replace(".csv", "_summary.csv"))
    print(f"Saved summary → {args.out_csv.replace('.csv', '_summary.csv')}")

    print("\n✅ Shopping-only baseline / domain-shift decomposition complete.")


if __name__ == "__main__":
    main()
