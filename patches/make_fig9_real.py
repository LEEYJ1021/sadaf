"""
make_fig9_real.py
==================
Run this AFTER re-running the patched scripts/09_robustness.py against the
real dataset, once figures/logo_cv_fold_rmse.csv exists (37 rows: one real
RMSE per held-out ad group, columns "group", "RMSE", "n_test").

This replaces the "illustrative distribution centered on reported mean+-SD"
version of Figure 9 with the actual 37 fold-level RMSE values -- a strip
plot is used rather than a histogram since n=37 is too small for a smooth
histogram to be a fair representation.

Usage:
    python make_fig9_real.py figures/logo_cv_fold_rmse.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "figures/logo_cv_fold_rmse.csv"
df = pd.read_csv(csv_path)

rmse = df["RMSE"].values
mean_rmse = rmse.mean()
sd_rmse = rmse.std()

fig, ax = plt.subplots(figsize=(9, 5.5))

rng = np.random.default_rng(0)
jitter = rng.uniform(-0.06, 0.06, size=len(rmse))
ax.scatter(rmse, jitter, s=55, alpha=0.75, color="#1f3b57", zorder=3,
           label=f"Real per-fold RMSE (n={len(rmse)})")

ax.axvline(mean_rmse, color="#b3401f", linewidth=2,
           label=f"LOGO-CV mean = {mean_rmse:.4f}")
ax.axvline(1.2099, color="#1f8a70", linestyle="--", linewidth=2,
           label="Group-split test RMSE = 1.2099 (LSTM)")
ax.axvspan(mean_rmse - sd_rmse, mean_rmse + sd_rmse, color="#b3401f", alpha=0.08,
           label=f"\u00b11 SD ({sd_rmse:.4f})")

ax.set_yticks([])
ax.set_xlabel("RMSE (log-ROAS)")
ax.set_title(
    "Figure 9 (corrected) \u2014 Leave-one-ad-group-out cross-validation\n"
    "Real per-fold RMSE, not a synthetic reconstruction"
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.savefig("fig9_logo_cv_real.png", dpi=200, bbox_inches="tight")
print(f"Saved fig9_logo_cv_real.png from {len(rmse)} real fold values "
      f"(mean={mean_rmse:.4f}, sd={sd_rmse:.4f})")
