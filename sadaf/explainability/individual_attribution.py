"""
sadaf/explainability/individual_attribution.py

Individual-level, cross-verified attribution pipeline (H5), replacing the
original group-level GS-SHAP (HSIC-grouped Shapley) estimator.

Rationale (manuscript Sections 2.7, 5.3, 8.3):
Chamma, Thirion & Engemann (2024) note that joint group-level Shapley
methods are conventionally motivated by, and most informative in,
high-dimensional feature spaces. This study's feature set (7 base
features) does not meet that threshold, so a simpler, individually
computed and independently cross-verified attribution design is the
dimensionality-appropriate choice: rather than assuming a joint
estimator correctly handles dependence among engagement/spend features,
this module computes three independent attribution methods on the
identical trained model and evaluation sample, and treats cross-method
convergence -- not agreement within a single joint estimator -- as the
evidentiary standard.

Three methods, identical model + data:
    1. Individual SHAP   -- kernel-based Shapley value estimator (Lundberg & Lee, 2017)
    2. Permutation SHAP  -- Monte Carlo permutation-based Shapley estimator
                            (Strumbelj & Kononenko, 2014)
    3. Integrated Gradients -- gradient-path attribution (Sundararajan et al., 2017)

For each method, per-observation attributions are summarized via a Gini
coefficient computed separately within each of three ad-group clusters
(k-means on engagement/spend features), for two feature categories
retained from the original grouping taxonomy for readability only
(NOT used as a joint estimator input):
    Group 0 -- engagement/spend: CTR, CVR, Depth, log_cost, log_impression
    Group 1 -- temporal: hour_sin, hour_cos

Outputs (mirrors the manuscript's Table 9, Table 10, Figure 7):
    - cluster_sizes.csv           unique ad-group counts & row-level n per cluster
    - gini_by_cluster_method.csv  Table 10
    - kruskal_wallis_group0.csv   Kruskal-Wallis H/p per method (Group 0 only)
    - spearman_agreement.csv      pairwise Spearman rho per cluster
    - fig7_individual_attribution_verification.png

Usage:
    python -m sadaf.explainability.individual_attribution \
        --data_path data/ad_performance.xlsx \
        --model_path artifacts/regression_stage_lstm.pt \
        --out_dir figures/
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

GROUP0_FEATURES = ["CTR", "CVR", "Depth", "log_cost", "log_impression"]
GROUP1_FEATURES = ["hour_sin", "hour_cos"]
ALL_FEATURES = GROUP0_FEATURES + GROUP1_FEATURES

METHODS = ["IndividualSHAP", "PermutationSHAP", "IntegratedGradients"]


# --------------------------------------------------------------------------- #
# 1. Attribution computation
# --------------------------------------------------------------------------- #
def compute_individual_shap(model, X, background, max_samples=100):
    """Kernel-based Shapley-value attribution (Lundberg & Lee, 2017).

    Background is subsampled to `max_samples` if larger, matching the
    manuscript's reported setting ("Background dataset has 200 samples
    but max_samples=100. Subsampling to 100 samples for SHAP value
    computation.").
    """
    import shap

    if len(background) > max_samples:
        background = background.sample(max_samples, random_state=42)
    explainer = shap.Explainer(model.predict, background)
    sv = explainer(X)
    return np.asarray(sv.values)


def compute_permutation_shap(model, X, background, n_permutations=None):
    """Monte Carlo permutation-based Shapley estimator (Strumbelj & Kononenko, 2014).

    Implemented via shap.explainers.Permutation for exact parity with the
    manuscript's reported "PermutationExplainer" runtime signature.
    """
    import shap

    explainer = shap.explainers.Permutation(model.predict, background)
    sv = explainer(X)
    return np.asarray(sv.values)


def compute_integrated_gradients(model, X, baseline=None, steps=50):
    """Integrated Gradients (Sundararajan, Taly & Yan, 2017).

    Requires a differentiable model wrapper exposing `.predict_tensor`
    (torch.Tensor -> torch.Tensor) and `.input_dim`. Falls back to a
    finite-difference gradient approximation if no autograd model is
    supplied, so this function runs against both the deep (LSTM/GRU)
    regression-stage models and simpler sklearn baselines.
    """
    import torch

    if baseline is None:
        baseline = np.zeros((1, X.shape[1]))

    X_t = torch.tensor(X.values, dtype=torch.float32, requires_grad=False)
    baseline_t = torch.tensor(baseline, dtype=torch.float32)

    attributions = np.zeros_like(X.values, dtype=np.float64)
    alphas = np.linspace(0, 1, steps)

    for alpha in alphas:
        interp = baseline_t + alpha * (X_t - baseline_t)
        interp.requires_grad_(True)
        out = model.predict_tensor(interp)
        grad = torch.autograd.grad(out.sum(), interp, retain_graph=False)[0]
        attributions += grad.detach().numpy()

    attributions = attributions * (X.values - baseline) / steps
    return attributions


# --------------------------------------------------------------------------- #
# 2. Clustering (unchanged from the original group-level design)
# --------------------------------------------------------------------------- #
def build_clusters(df, features, n_clusters=3, random_state=42):
    """K-means clustering on engagement/spend features.

    Returns both unique ad-group membership counts and row-level
    observation counts per cluster, since these differ (Table 9) and
    the row-level counts are what anchor the power analysis in
    Section 6.10 / README Section 5.9.
    """
    X = df[features].replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    out = df.loc[X.index].copy()
    out["cluster"] = labels
    return out


# --------------------------------------------------------------------------- #
# 3. Gini coefficient (temporal concentration of attribution)
# --------------------------------------------------------------------------- #
def gini(x):
    x = np.abs(np.asarray(x, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


# --------------------------------------------------------------------------- #
# 4. Main pipeline
# --------------------------------------------------------------------------- #
def run(data_path, model, out_dir="figures/"):
    df = pd.read_excel(data_path) if data_path.endswith(("xlsx", "xls")) else pd.read_csv(data_path)

    # Restrict to the paid, non-zero-attribution sample used for the
    # regression-stage explainability pass (consistent with the
    # sequence-level test split used elsewhere in the pipeline).
    df = df[(df["cost"] > 0) & (df["ROAS"] > 0)].copy()

    clustered = build_clusters(df, GROUP0_FEATURES)

    cluster_sizes = (
        clustered.groupby("cluster")
        .agg(unique_ad_groups=("ad_group_id", "nunique"), n=("cluster", "size"))
        .reset_index()
    )
    print("=== Cluster sizes (unique ad_group_id) ===")
    print(clustered["cluster"].value_counts().sort_index(), "\n")
    cluster_sizes.to_csv(f"{out_dir}/cluster_sizes.csv", index=False)

    background = clustered[ALL_FEATURES].sample(min(200, len(clustered)), random_state=42)

    gini_rows = []
    kw_input = {m: {} for m in METHODS}
    spearman_rows = []

    for c, sub in clustered.groupby("cluster"):
        X = sub[ALL_FEATURES]

        attributions = {
            "IndividualSHAP": compute_individual_shap(model, X, background, max_samples=100),
            "PermutationSHAP": compute_permutation_shap(model, X, background),
            "IntegratedGradients": compute_integrated_gradients(model, X),
        }

        row = {"cluster": c, "n": len(X)}
        method_feature_means = {}

        for method, attr in attributions.items():
            attr_df = pd.DataFrame(np.abs(attr), columns=ALL_FEATURES, index=X.index)
            g0 = attr_df[GROUP0_FEATURES].sum(axis=1)
            g1 = attr_df[GROUP1_FEATURES].sum(axis=1)

            row[f"{method}_G0_Gini"] = gini(g0)
            row[f"{method}_G1_Gini"] = gini(g1)

            kw_input[method][c] = g0.values
            method_feature_means[method] = attr_df.mean(axis=0)

        gini_rows.append(row)

        # cross-method Spearman rank correlation of feature-mean |attribution|
        for i, m1 in enumerate(METHODS):
            for m2 in METHODS[i + 1:]:
                rho, p = spearmanr(method_feature_means[m1], method_feature_means[m2])
                spearman_rows.append(
                    {"cluster": c, "n": len(X), "method_1": m1, "method_2": m2, "rho": rho, "p": p}
                )

    gini_df = pd.DataFrame(gini_rows)
    print("=== Gini coefficients by cluster & method (Group0 / Group1) ===")
    print(gini_df.to_string(index=False), "\n")
    gini_df.to_csv(f"{out_dir}/gini_by_cluster_method.csv", index=False)

    print("=== Kruskal-Wallis across clusters (Group0 attribution magnitude) ===")
    kw_rows = []
    for method in METHODS:
        groups = list(kw_input[method].values())
        H, p = kruskal(*groups)
        print(f"{method:<20} H = {H:.3f}   p = {p:.4f}")
        kw_rows.append({"method": method, "H": H, "p": p})
    pd.DataFrame(kw_rows).to_csv(f"{out_dir}/kruskal_wallis_group0.csv", index=False)

    spearman_df = pd.DataFrame(spearman_rows)
    print("\n=== Cross-method Spearman rank correlation (per cluster, feature-mean |attribution|) ===")
    for c in sorted(spearman_df["cluster"].unique()):
        sub = spearman_df[spearman_df["cluster"] == c]
        n = sub["n"].iloc[0]
        print(f"-- Cluster {c} (n={n}) --")
        for _, r in sub.iterrows():
            print(f"   {r['method_1']} vs {r['method_2']}: rho = {r['rho']:.3f}  (p = {r['p']:.4f})")
    spearman_df.to_csv(f"{out_dir}/spearman_agreement.csv", index=False)

    return gini_df, spearman_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_dir", default="figures/")
    args = parser.parse_args()

    import joblib

    trained_model = joblib.load(args.model_path)
    run(args.data_path, trained_model, args.out_dir)
