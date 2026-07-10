"""
PATCH — scripts/09_robustness.py :: main()
=============================================
FIX-24: run_logo_cv()'s return value (37 rows: one real RMSE per held-out
ad group) was previously computed but never captured or saved -- main()
called `run_logo_cv(df_roas)` and discarded the return value, so only the
printed mean +/- SD survived. Figure 9 in the paper/README was then built
as an "illustrative distribution centered on reported mean+/-SD" because
the real per-fold values no longer existed anywhere.

This patch captures the DataFrame and writes it to
figures/logo_cv_fold_rmse.csv so Figure 9 can be redrawn from the 37 real
values instead of a synthetic reconstruction.

Apply by replacing this single line in main():

    run_logo_cv(df_roas)

with:

    logo_df = run_logo_cv(df_roas)
    logo_df.to_csv("figures/logo_cv_fold_rmse.csv", index=False)
    print(f"  [FIX-24] Saved {len(logo_df)} per-fold RMSE values to "
          f"figures/logo_cv_fold_rmse.csv")

No other change is required -- run_logo_cv() already builds and returns
this DataFrame (see logo_rmses / logo_df in the existing function body),
it just needs to be kept instead of discarded.
"""

# --- full replacement of the relevant section of main() for convenience ---
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="SADAF: robustness checks (LOGO-CV, reg grid, DM correction)")
#     parser.add_argument("--data_path", type=str,
#                          default="data/3월성과데이터(샘플).xlsx")
#     parser.add_argument("--target_n", type=int, default=800)
#     args = parser.parse_args()
#
#     df, df_paid, df_roas = load_and_preprocess(args.data_path)
#
#     # [FIX-24] capture and persist per-fold LOGO-CV RMSE values
#     logo_df = run_logo_cv(df_roas)
#     import os
#     os.makedirs("figures", exist_ok=True)
#     logo_df.to_csv("figures/logo_cv_fold_rmse.csv", index=False)
#     print(f"  [FIX-24] Saved {len(logo_df)} per-fold RMSE values to "
#           f"figures/logo_cv_fold_rmse.csv")
#
#     ... (rest of main() unchanged)
