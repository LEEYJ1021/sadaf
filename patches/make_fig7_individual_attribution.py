"""
patches/make_fig7_individual_attribution.py

Regenerates Figure 7 ("Individual-Attribution Verification") from the
CSV outputs of sadaf/explainability/individual_attribution.py.

Replaces the original group-level GS-SHAP figure (old fig4_gsshap_gini.png)
now that the explainability pillar has moved to individual-level,
cross-verified attribution (Individual SHAP, Permutation SHAP, Integrated
Gradients) per manuscript Sections 2.7, 5.3, 6.7.

Panel layout (2x2):
    top-left     Group 0 (engagement/spend) attribution concentration, by cluster & method
    top-right    Group 1 (temporal: hour sin/cos) attribution concentration, by cluster & method
    bottom-left  Kruskal-Wallis H statistic per method (Group 0 attribution differs across clusters)
    bottom-right Cross-method Spearman rank correlation, by cluster

Usage:
    python patches/make_fig7_individual_attribution.py \
        --gini figures/gini_by_cluster_method.csv \
        --kw figures/kruskal_wallis_group0.csv \
        --spearman figures/spearman_agreement.csv \
        --out assets/fig7_individual_attribution_verification.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["IndividualSHAP", "PermutationSHAP", "IntegratedGradients"]
METHOD_COLORS = {"IndividualSHAP": "#3874A6", "PermutationSHAP": "#D97E28", "IntegratedGradients": "#4A9159"}
METHOD_LABELS = {"IndividualSHAP": "Ind.SHAP", "PermutationSHAP": "Perm.SHAP", "IntegratedGradients": "IG"}

# Fallback values matching the manuscript's Table 10 / §6.7 (used only if
# no CSV outputs are supplied — e.g., for a documentation-only re-render).
DEFAULT_GINI = pd.DataFrame([
    {"cluster": 0, "n": 1214, "IndividualSHAP_G0_Gini": 0.368194, "IndividualSHAP_G1_Gini": 0.648485,
     "PermutationSHAP_G0_Gini": 0.425272, "PermutationSHAP_G1_Gini": 0.681121,
     "IntegratedGradients_G0_Gini": 0.129057, "IntegratedGradients_G1_Gini": 0.242154},
    {"cluster": 1, "n": 217, "IndividualSHAP_G0_Gini": 0.365977, "IndividualSHAP_G1_Gini": 0.543975,
     "PermutationSHAP_G0_Gini": 0.333969, "PermutationSHAP_G1_Gini": 0.594746,
     "IntegratedGradients_G0_Gini": 0.099013, "IntegratedGradients_G1_Gini": 0.269950},
    {"cluster": 2, "n": 28, "IndividualSHAP_G0_Gini": 0.317709, "IndividualSHAP_G1_Gini": 0.429084,
     "PermutationSHAP_G0_Gini": 0.286485, "PermutationSHAP_G1_Gini": 0.451455,
     "IntegratedGradients_G0_Gini": 0.091972, "IntegratedGradients_G1_Gini": 0.392960},
])

DEFAULT_KW = pd.DataFrame([
    {"method": "IndividualSHAP", "H": 72.213, "p": 0.0000},
    {"method": "PermutationSHAP", "H": 143.831, "p": 0.0000},
    {"method": "IntegratedGradients", "H": 18.277, "p": 0.0001},
])

DEFAULT_SPEARMAN = pd.DataFrame([
    {"cluster": 0, "n": 1214, "method_1": "IndividualSHAP", "method_2": "PermutationSHAP", "rho": 0.857, "p": 0.0137},
    {"cluster": 0, "n": 1214, "method_1": "IndividualSHAP", "method_2": "IntegratedGradients", "rho": 0.607, "p": 0.1482},
    {"cluster": 0, "n": 1214, "method_1": "PermutationSHAP", "method_2": "IntegratedGradients", "rho": 0.821, "p": 0.0234},
    {"cluster": 1, "n": 217, "method_1": "IndividualSHAP", "method_2": "PermutationSHAP", "rho": 0.964, "p": 0.0005},
    {"cluster": 1, "n": 217, "method_1": "IndividualSHAP", "method_2": "IntegratedGradients", "rho": 0.714, "p": 0.0713},
    {"cluster": 1, "n": 217, "method_1": "PermutationSHAP", "method_2": "IntegratedGradients", "rho": 0.821, "p": 0.0234},
    {"cluster": 2, "n": 28, "method_1": "IndividualSHAP", "method_2": "PermutationSHAP", "rho": 0.964, "p": 0.0005},
    {"cluster": 2, "n": 28, "method_1": "IndividualSHAP", "method_2": "IntegratedGradients", "rho": 0.821, "p": 0.0234},
    {"cluster": 2, "n": 28, "method_1": "PermutationSHAP", "method_2": "IntegratedGradients", "rho": 0.893, "p": 0.0068},
])


def plot(gini_df, kw_df, spearman_df, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Individual-Attribution Verification", fontsize=16, fontweight="bold")

    clusters = sorted(gini_df["cluster"].unique())
    cluster_labels = [f"Cluster {c}\n(n={gini_df.loc[gini_df.cluster == c, 'n'].iloc[0]})" for c in clusters]
    x = np.arange(len(clusters))
    width = 0.25

    # --- top-left: Group 0 concentration ---
    ax = axes[0, 0]
    for i, m in enumerate(METHODS):
        vals = [gini_df.loc[gini_df.cluster == c, f"{m}_G0_Gini"].iloc[0] for c in clusters]
        ax.bar(x + (i - 1) * width, vals, width, label=m, color=METHOD_COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels)
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Group 0 (engagement/spend) attribution concentration", fontweight="bold")
    ax.legend()

    # --- top-right: Group 1 concentration ---
    ax = axes[0, 1]
    for i, m in enumerate(METHODS):
        vals = [gini_df.loc[gini_df.cluster == c, f"{m}_G1_Gini"].iloc[0] for c in clusters]
        ax.bar(x + (i - 1) * width, vals, width, label=m, color=METHOD_COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels)
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Group 1 (temporal: hour sin/cos) attribution concentration", fontweight="bold")
    ax.legend()

    # --- bottom-left: Kruskal-Wallis H by method ---
    ax = axes[1, 0]
    bars = ax.bar(kw_df["method"], kw_df["H"], color=[METHOD_COLORS[m] for m in kw_df["method"]])
    for bar, p in zip(bars, kw_df["p"]):
        label = "p<.0001" if p < 0.0001 else f"p={p:.4f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, label,
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Kruskal-Wallis H statistic")
    ax.set_title("Group 0 attribution differs across clusters (by method)", fontweight="bold")

    # --- bottom-right: cross-method Spearman agreement ---
    ax = axes[1, 1]
    pair_labels = ["Ind.SHAP\nvs Perm.SHAP", "Ind.SHAP\nvs IG", "Perm.SHAP\nvs IG"]
    pair_keys = [("IndividualSHAP", "PermutationSHAP"), ("IndividualSHAP", "IntegratedGradients"),
                 ("PermutationSHAP", "IntegratedGradients")]
    colors = ["#3874A6", "#D97E28", "#4A9159"]
    for c, color in zip(clusters, colors):
        sub = spearman_df[spearman_df.cluster == c]
        n = sub["n"].iloc[0]
        vals = []
        for m1, m2 in pair_keys:
            row = sub[((sub.method_1 == m1) & (sub.method_2 == m2)) | ((sub.method_1 == m2) & (sub.method_2 == m1))]
            vals.append(row["rho"].iloc[0])
        ax.plot(pair_labels, vals, marker="o", linewidth=2, markersize=8, color=color, label=f"Cluster {c} (n={n})")
    ax.axhline(0.8, linestyle="--", color="gray", linewidth=1)
    ax.text(2.05, 0.8, "conventional 'strong agreement'\nthreshold (\u03c1=0.8)", fontsize=8, color="gray", va="center")
    ax.set_ylabel("Spearman rank correlation (\u03c1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-method attribution agreement, by cluster", fontweight="bold")
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gini", default=None)
    parser.add_argument("--kw", default=None)
    parser.add_argument("--spearman", default=None)
    parser.add_argument("--out", default="assets/fig7_individual_attribution_verification.png")
    args = parser.parse_args()

    gini_df = pd.read_csv(args.gini) if args.gini else DEFAULT_GINI
    kw_df = pd.read_csv(args.kw) if args.kw else DEFAULT_KW
    spearman_df = pd.read_csv(args.spearman) if args.spearman else DEFAULT_SPEARMAN

    plot(gini_df, kw_df, spearman_df, args.out)
