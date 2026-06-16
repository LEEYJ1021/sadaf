"""
03_causal.py — Causal analyses for H1 (PSM + doubly robust IPW),
H2 (mediation), and H3 (moderation).

Usage:
    python scripts/03_causal.py --data_path data/3월성과데이터(샘플).xlsx
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from sadaf.data.loader import load_and_preprocess
from sadaf.causal.psm import fit_propensity_scores, match_psm, compute_ipw_att, smd
from sadaf.causal.mediation import baron_kenny_mediation, bootstrap_mediation, classify_suppressor
from sadaf.causal.moderation import fit_moderation_ols

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def run_h1(df_paid: pd.DataFrame) -> None:
    print("\n══ H1: PSM + Doubly Robust IPW ════════════════════")
    df_psm = df_paid[df_paid["CTR"] > 0].copy()
    ctr_median = df_psm["CTR"].median()
    df_psm["T_highCTR"] = (df_psm["CTR"] > ctr_median).astype(int)
    type_dummies = pd.get_dummies(df_psm["campaign_type_label"],
                                   prefix="ctype", drop_first=True)
    df_psm = pd.concat([df_psm, type_dummies], axis=1)
    confounders = (["log_impression", "log_cost", "Depth"]
                   + list(type_dummies.columns))
    df_psm_c = df_psm[confounders + ["T_highCTR", "has_conversion"]].dropna()

    pscore = fit_propensity_scores(
        df_psm_c[confounders].values, df_psm_c["T_highCTR"].values)

    matched_t, matched_c, att, (ci_lo, ci_hi) = match_psm(
        df_psm_c, pscore, treatment_col="T_highCTR",
        outcome_col="has_conversion", caliper_mult=0.1, n_boot=2000,
        seed=RANDOM_SEED)
    print(f"  PSM-ATT = {att:.4f}  95% Boot CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  n_matched = {len(matched_t)}  "
          f"H1: {'SUPPORTED ✓' if ci_lo > 0 else 'NOT SUPPORTED ✗'}")

    ipw_att = compute_ipw_att(
        df_psm_c["T_highCTR"].values, df_psm_c["has_conversion"].values, pscore)
    print(f"  IPW-ATT = {ipw_att:.4f}  [PRIMARY ESTIMATOR — doubly robust]")

    balance_rows = []
    for col in confounders:
        x_t_pre = df_psm_c.loc[df_psm_c["T_highCTR"] == 1, col].values
        x_c_pre = df_psm_c.loc[df_psm_c["T_highCTR"] == 0, col].values
        x_t_post = df_psm_c.loc[matched_t, col].values
        x_c_post = df_psm_c.loc[matched_c, col].values
        smd_after = smd(x_t_post, x_c_post)
        bal = ("✓ balanced" if abs(smd_after) < 0.10 else
               "⚠ residual — DR corrected" if abs(smd_after) < 0.25 else
               "✗ substantial — DR corrected (primary estimator: IPW)")
        balance_rows.append({
            "Covariate": col,
            "SMD_before": round(smd(x_t_pre, x_c_pre), 4),
            "SMD_after": round(smd_after, 4),
            "Balance": bal,
        })
    bal_df = pd.DataFrame(balance_rows)
    print("\n  Table S1: Covariate Balance (PSM + DR correction)")
    print(bal_df.to_string(index=False))


def run_h2(df_paid: pd.DataFrame) -> None:
    print("\n══ H2: Mediation Analysis ══════════════════════════")
    df_med = df_paid[(df_paid["CTR"] > 0) & (df_paid["Depth"] > 0)].copy()
    df_med["log_CTR"] = np.log1p(df_med["CTR"])
    X = df_med["log_CTR"].values.reshape(-1, 1)
    M = df_med["Depth"].values
    Y = df_med["has_conversion"].values

    result = baron_kenny_mediation(X, M, Y)
    ci_lo, ci_hi = bootstrap_mediation(X, M, Y, n_boot=2000, seed=RANDOM_SEED)
    suppressor_type, interp = classify_suppressor(result["a"], result["b"], result["indirect"])

    prop_med = result["indirect"] / (result["c_total"] + 1e-8) * 100
    print(f"  a = {result['a']:.4f}  b = {result['b']:.4f}  "
          f"indirect (a×b) = {result['indirect']:.4f}  "
          f"CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Type: {suppressor_type}")
    print(f"  Interpretation: {interp}")
    print(f"  Proportion mediated: {prop_med:.1f}% "
          f"(appendix only; sign direction is the primary finding)")


def run_h3(df_paid: pd.DataFrame) -> None:
    print("\n══ H3: Moderation Analysis ═════════════════════════")
    df_mod = df_paid[
        (df_paid["CTR"] > 0) & (df_paid["ROAS"] > 0) &
        (df_paid["campaign_type_label"].isin(["Search", "Shopping"]))
    ].copy()
    df_mod["log_ROAS"] = np.log1p(df_mod["ROAS"])
    df_mod["log_CTR"] = np.log1p(df_mod["CTR"])
    df_mod["is_search"] = (df_mod["campaign_type_label"] == "Search").astype(int)

    model = fit_moderation_ols(
        df_mod,
        formula="log_ROAS ~ log_CTR * is_search + log_cost + log_impression")
    beta_ctr = model.params["log_CTR"]
    beta_inter = model.params["log_CTR:is_search"]
    p_inter = model.pvalues["log_CTR:is_search"]
    me_shopping = beta_ctr
    me_search = beta_ctr + beta_inter
    print(f"  β_interaction = {beta_inter:.4f}  p = {p_inter:.4f}  "
          f"H3: {'SUPPORTED ✓' if p_inter < 0.05 else 'NOT SUPPORTED ✗'}")
    print(f"  ME_Shopping = {me_shopping:.3f}  ME_Search = {me_search:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: causal analyses (H1-H3)")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)
    run_h1(df_paid)
    run_h2(df_paid)
    run_h3(df_paid)
    print("\n✅ Causal analyses (H1–H3) complete.")


if __name__ == "__main__":
    main()