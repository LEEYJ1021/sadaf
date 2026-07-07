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
from sadaf.causal.psm import run_psm_ipw  # [FIX-11]
from sadaf.causal.mediation import run_mediation  # [FIX-12]
from sadaf.causal.moderation import run_moderation  # [FIX-13]

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def run_h1(df_paid: pd.DataFrame) -> None:
    """[FIX-11] Calls the current sadaf.causal.psm API (run_psm_ipw),
    which handles propensity scoring, caliper matching, bootstrap CI,
    IPW-ATT, and the balance table internally. IPW-ATT remains the
    PRIMARY doubly-robust estimator; PSM-ATT is corroborating only."""
    print("\n══ H1: PSM + Doubly Robust IPW [FIX-11] ═══════════")
    result = run_psm_ipw(df_paid, random_seed=RANDOM_SEED)
    print(f"  PSM-ATT = {result['psm_att']:.4f}  "
          f"95% Boot CI = [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]")
    print(f"  n_matched = {result['n_matched']}  "
          f"H1 (PSM, corroborating): "
          f"{'SUPPORTED ✓' if result['h1_supported'] else 'NOT SUPPORTED ✗'}")
    print(f"  IPW-ATT = {result['ipw_att']:.4f}  [PRIMARY ESTIMATOR — doubly robust]")
    print(f"  DR consistency (|IPW-ATT − PSM-ATT| < 0.05): "
          f"{'✓ consistent' if result['dr_consistent'] else '⚠ divergent — inspect balance table'}")
    print("\n  Table S1: Covariate Balance (PSM + DR correction)")
    print(result["balance_df"].to_string(index=False))


def run_h2(df_paid: pd.DataFrame) -> None:
    """[FIX-12] Calls the current sadaf.causal.mediation API
    (run_mediation), which handles path-a/path-b estimation,
    bootstrap CI on the indirect path, and suppressor classification
    internally."""
    print("\n══ H2: Mediation Analysis [FIX-12] ══════════════════")
    result = run_mediation(df_paid, random_seed=RANDOM_SEED)
    print(f"  a = {result['a']:.4f}  b = {result['b']:.4f}  "
          f"indirect (a×b) = {result['indirect']:.4f}  "
          f"CI = [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]")
    print(f"  Type: {result['suppressor_type']}")
    print(f"  Interpretation: {result['suppressor_interp']}")
    print(f"  Proportion mediated: {result['prop_mediated']:.1f}% "
          f"(appendix only; sign direction is the primary finding)")
    print(f"  H2: {'SUPPORTED ✓' if result['h2_supported'] else 'NOT SUPPORTED ✗'} "
          f"(bootstrap CI excludes 0)")


def run_h3(df_paid: pd.DataFrame) -> None:
    """[FIX-13] Calls the current sadaf.causal.moderation API
    (run_moderation), which fits the HC3-robust OLS interaction model
    and returns marginal effects directly."""
    print("\n══ H3: Moderation Analysis [FIX-13] ═════════════════")
    result = run_moderation(df_paid)
    print(f"  β_interaction = {result['beta_interaction']:.4f}  "
          f"p = {result['p_interaction']:.4f}  "
          f"H3: {'SUPPORTED ✓' if result['h3_supported'] else 'NOT SUPPORTED ✗'}")
    print(f"  ME_Shopping = {result['me_shopping']:.3f}  "
          f"ME_Search = {result['me_search']:.3f}")
    print(f"  R² = {result['r_squared']:.4f}  n = {result['n_obs']}")


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