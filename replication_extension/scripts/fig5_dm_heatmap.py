"""
fig5_dm_heatmap.py

Rebuilds Figure 5 ("Pairwise Diebold-Mariano test results") -- Section 6.5.

Fix applied: the previous version left three cells (Mamba-Ridge, Mamba-MLP,
Ridge-MLP) blank/uncomputed, which silently reduced the effective comparison
count from 21 (=7 choose 2) to 18 and produced a mismatch with the
manuscript text's "14/21 significant before correction, 8/21 after FDR"
(Section 6.5, matching 05_prediction.log's own summary line: "14/21 pairs
significant at raw p<0.05; 8/21 remain significant after BH-FDR"). This
version fills in the full 7x7 = 21-pair matrix using the exact DM/p values
reported in 05_prediction.log, so the heatmap and the manuscript text are
numerically consistent.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"]
mpl.rcParams["axes.unicode_minus"] = False

archs = ["BayesianLSTM", "LSTM", "GRU", "BiLSTM", "Mamba", "Ridge", "MLP"]
n = len(archs)

# Exact values from 05_prediction.log ("DM Comparisons: raw p + BH-FDR correction [FIX-10]")
# key: frozenset({a, b}) -> (p_raw, p_fdr)
pairs = {
    ("BayesianLSTM", "LSTM"):   (0.0042, 0.0293),
    ("BayesianLSTM", "GRU"):    (0.2692, 0.3140),
    ("BayesianLSTM", "BiLSTM"): (0.1122, 0.1472),
    ("BayesianLSTM", "Mamba"):  (0.0255, 0.0537),
    ("BayesianLSTM", "Ridge"):  (0.0410, 0.0615),
    ("BayesianLSTM", "MLP"):    (0.0077, 0.0316),
    ("LSTM", "GRU"):            (0.0105, 0.0316),
    ("LSTM", "BiLSTM"):         (0.0105, 0.0316),
    ("LSTM", "Mamba"):          (0.0024, 0.0251),
    ("LSTM", "Ridge"):          (0.0078, 0.0316),
    ("LSTM", "MLP"):            (0.0021, 0.0251),
    ("GRU", "BiLSTM"):          (0.1254, 0.1549),
    ("GRU", "Mamba"):           (0.0283, 0.0540),
    ("GRU", "Ridge"):           (0.0239, 0.0537),
    ("GRU", "MLP"):             (0.0125, 0.0329),
    ("BiLSTM", "Mamba"):        (0.0381, 0.0615),
    ("BiLSTM", "Ridge"):        (0.0369, 0.0615),
    ("BiLSTM", "MLP"):          (0.0561, 0.0786),
    ("Mamba", "Ridge"):         (0.6763, 0.6763),
    ("Mamba", "MLP"):           (0.5628, 0.5909),
    ("Ridge", "MLP"):           (0.3568, 0.3944),
}
assert len(pairs) == 21

def get_pair(a, b):
    if (a, b) in pairs:
        return pairs[(a, b)]
    return pairs[(b, a)]

raw_mat = np.full((n, n), np.nan)
fdr_mat = np.full((n, n), np.nan)
for i, a in enumerate(archs):
    for j, b in enumerate(archs):
        if i == j:
            continue
        p_raw, p_fdr = get_pair(a, b)
        raw_mat[i, j] = p_raw
        fdr_mat[i, j] = p_fdr

def plot_heatmap(ax, mat, title):
    # Diverging color: low p (significant) = green, high p (n.s.) = red
    cmap = plt.get_cmap("RdYlGn_r")
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(archs, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticklabels(archs, fontsize=9.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="#999999", linewidth=0.5))
                continue
            val = mat[i, j]
            star = "*" if val < 0.05 else ""
            txt_color = "white" if (val < 0.15 or val > 0.75) else "black"
            ax.text(j, i, f"{val:.3f}{star}", ha="center", va="center",
                    fontsize=8.2, color=txt_color)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.5)
    ax.grid(False)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3), dpi=300)
plot_heatmap(axes[0], raw_mat, "Raw p-value")
plot_heatmap(axes[1], fdr_mat, "BH-FDR corrected p-value")

fig.suptitle(
    "Diebold\u2013Mariano pairwise model comparison \u2014 regression stage (test n = 24)\n"
    "All 21 of 21 pairwise comparisons shown \u2014 14/21 significant raw (p<.05), 8/21 significant after BH-FDR",
    fontsize=12.5, fontweight="bold", y=1.03
)

plt.tight_layout()
out_path = "/mnt/user-data/outputs/assets/fig5_dm_heatmap.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", out_path)

# Sanity check counts against the log's own summary line
raw_sig = sum(1 for v in pairs.values() if v[0] < 0.05)
fdr_sig = sum(1 for v in pairs.values() if v[1] < 0.05)
print(f"raw significant: {raw_sig}/21  (log states 14/21)")
print(f"FDR significant: {fdr_sig}/21  (log states 8/21)")
