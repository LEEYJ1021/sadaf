"""
fig10_power_summary.py

Regenerates Figure 10 ("Statistical power by hypothesis and diagnostic
test") -- Section 6.10 / Table 12 of the manuscript.

Fix applied: the previous version of this figure still showed the
pre-redesign H5 entry as a single row ("H5 Kruskal-Wallis, n=24, 3
clusters, power=.53") and labeled the H5 Spearman clusters as n=7/8/9.
This directly contradicted Table 12, which reports THREE separate
Kruskal-Wallis rows (one per attribution method) computed on the
row-level sample (n=1,459 pooled across 3 clusters: 1,214+217+28), with
power >.99 / >.99 / .93 respectively -- all comfortably at or above the
0.80 threshold, not underpowered as the old figure implied. The Spearman
row is updated to match Table 12's explicit entry: Cluster 2 (n=28),
weakest pair (Individual SHAP vs. Integrated Gradients, rho=0.607),
power=.59.

Note on a residual manuscript inconsistency (flagged, not silently
"fixed"): Table 12 labels this weakest Spearman pair as belonging to
"Cluster 2 (n=28)", while the narrative prose in Section 6.7/6.10
describes the same rho=0.607 value as occurring "in the larger Cluster
0" (n=1,214). This figure follows Table 12's explicit row exactly, since
that is the structured source for this power summary -- but the
underlying cluster-label discrepancy between Table 12 and the Section
6.7/6.10 narrative should be reconciled in the manuscript text itself
before submission.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

rows = [
    ("H1 causal ATT\n(n = 14,987)", 0.99, "power"),
    ("H3 moderated OLS\n(n = 9,069)", 0.99, "power"),
    ("H4b DM pairwise, raw \u03b1\n(n = 24, mean of 18 pairs)", 0.64, "power"),
    ("H4b DM pairwise, FDR-equiv.\n(n = 24, mean of 18 pairs)", 0.24, "power"),
    ("H5 Kruskal-Wallis, Individual SHAP\n(n = 1,459 row-level, 3 clusters)", 0.99, "power"),
    ("H5 Kruskal-Wallis, Permutation SHAP\n(n = 1,459 row-level, 3 clusters)", 0.99, "power"),
    ("H5 Kruskal-Wallis, Integrated Gradients\n(n = 1,459 row-level, 3 clusters)", 0.93, "power"),
    ("H5 Spearman, Cluster 2 (n = 28)\nweakest pair (Ind.SHAP vs. IG, \u03c1=0.607)", 0.59, "power"),
    ("RQ6 LOAO-CV precision\n(n = 37 advertiser folds)", 0.95, "precision"),
]

labels = [r[0] for r in rows]
values = [r[1] for r in rows]
kinds = [r[2] for r in rows]

colors = []
for v, k in zip(values, kinds):
    if k == "precision":
        colors.append("#1f7a5c")
    elif v >= 0.80:
        colors.append("#2f8f6f")
    else:
        colors.append("#c1440e")

fig, ax = plt.subplots(figsize=(11, 7))
y_pos = range(len(labels))

bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.62)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlim(0, 1.08)
ax.set_xlabel("Statistical power (or precision proxy for RQ6 LOAO-CV)", fontsize=11)
ax.axvline(0.80, color="gray", linestyle="--", linewidth=1)
ax.text(0.805, -0.7, "conventional 0.80\npower threshold", fontsize=8, color="gray")

for bar, v in zip(bars, values):
    ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}", va="center", fontsize=9.5)

ax.set_title(
    "Figure 10. Statistical power by hypothesis and diagnostic test\n"
    "(H5 rows reflect the row-level, cross-verified attribution design; see Table 12)",
    fontsize=12, fontweight="bold",
)
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
out_path = "/mnt/user-data/outputs/assets/fig10_power_summary.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
print("Saved:", out_path)
