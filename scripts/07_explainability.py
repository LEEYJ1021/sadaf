"""
07_explainability.py  [FIXED v2 — patch applied]
----------------------------------
Multi-method attribution comparison (H5):
GS-SHAP (primary), Integrated Gradients, Permutation SHAP, Attention.

Changes vs. original
---------------------
FIX-2  (Data Leakage):
  - build_sequences() now returns group_ids; group_time_split() used for
    the regression/explainability split.

FIX-3  (Temporal Gini blank in Figure 9):
  - GS-SHAP loop calls explainer.explain_with_gini() and collects the
    returned gini arrays per cluster.
  - cell_maps_by_cluster dict is populated for compute_cluster_gini().
  - Both feat_imp_gs (mean |attribution|) AND gini_by_cluster (Gini per
    sample per feature) are produced and printed.
  - A plot_figure9() function generates the corrected Figure 9 with a
    populated right panel.

PATCH (2025-06):
  - REMOVED duplicate/wrong import:
      from sadaf.training.trainer import train_model, eval_reg, SeqDataset
    SeqDataset is correctly imported from sadaf.data.sequence (line 44).
    trainer only exports train_model + eval_reg.
  - FIXED ref_lstm= → ref_model= in augment_pipeline() call (line ~147).

Usage:
    python scripts/07_explainability.py --data_path data/3월성과데이터(샘플).xlsx
"""
import argparse
import importlib.util
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader

from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import (build_sequences, time_split,
                                  group_time_split,           # [FIX-2]
                                  normalize_sequences as normalize_X,
                                  SeqDataset)
from sadaf.augmentation.pipeline import augment_pipeline
from sadaf.models.gru import GRUForecaster
from sadaf.models.lstm import LSTMForecaster
from sadaf.models.attention import LSTMWithAttention
from sadaf.explainability.gsshap import (GSSHAP,
                                          compute_cluster_gini)  # [FIX-3]
from sadaf.explainability.intgrad import integrated_gradients
from sadaf.explainability.permshap import permutation_shap
# PATCH: removed wrong 'SeqDataset' import from trainer — it lives in sadaf.data.sequence
from sadaf.training.trainer import train_model, eval_reg

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES       = ["CTR", "CVR", "Depth", "log_cost",
                  "log_impression", "hour_sin", "hour_cos"]
D_IN           = len(FEATURES)
N_EXPLAIN      = 20
CLUSTER_NAMES  = ["C0 High-Volume", "C1 High-Conversion", "C2 Click-Rich"]


def load_gsshap_explainer(model, X_train):
    """Load GSSHAP — prefers the fixed module in sadaf.explainability.gsshap."""
    try:
        from sadaf.explainability.gsshap import GSSHAP
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            "gsshap", "sadaf/explainability/gsshap_standalone.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        GSSHAP = mod.GSSHAP
    return GSSHAP(
        model=model, X_train=X_train, task="reg", device=DEVICE,
        hsic_max_samples=2000, min_seg_len=2, max_segments=4,
        threshold_permutations=30, num_permutations=100, batch_size=64)


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-3] Figure 9 plotting with corrected Gini right panel
# ─────────────────────────────────────────────────────────────────────────────
def plot_figure9(
    feat_imp_gs: dict,
    gini_by_cluster: dict,
    features: list,
    cluster_names: list,
    sig_labels: dict,           # {feature_name: label_str}  e.g. {"CTR": "*", ...}
    out_path: str = "figures/fig_09_gsshap_importance_fixed.png",
):
    """
    [FIX-3] Reproduce Figure 9 with a properly populated Temporal Gini panel.

    Left panel  — GS-SHAP |attribution| boxplots by cluster (unchanged logic).
    Right panel — Temporal Gini boxplots by cluster (FIXED: uses abs values).
    """
    colors = ["#4C72B0", "#DD8452", "#55A868"]   # C0, C1, C2
    n_feat = len(features)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle("Figure 9. GS-SHAP Feature Importances & Temporal Concentration"
                 " (RQ5 / H5) [FIXED]", fontweight="bold")

    # ── Left: |attribution| boxplots ──────────────────────────────────────
    ax = axes[0]
    ax.set_title("GS-SHAP attribution by cluster")
    positions_per_feat = np.arange(n_feat)
    width = 0.22
    offsets = [-width, 0, width]

    for ci, cname in enumerate(cluster_names):
        data_mat = feat_imp_gs.get(ci, np.zeros((1, n_feat)))  # (n_samples, D)
        for fi in range(n_feat):
            col_data = data_mat[:, fi]
            ax.boxplot(
                col_data,
                positions=[positions_per_feat[fi] + offsets[ci]],
                widths=width * 0.85,
                patch_artist=True,
                medianprops=dict(color="orange", linewidth=1.5),
                boxprops=dict(facecolor=colors[ci], alpha=0.7),
                whiskerprops=dict(color=colors[ci]),
                capprops=dict(color=colors[ci]),
                flierprops=dict(marker="o", markersize=2,
                                markerfacecolor=colors[ci], alpha=0.4),
                showfliers=True,
            )

    # significance annotations
    for fi, feat in enumerate(features):
        label = sig_labels.get(feat, "ns")
        y_top = max(
            feat_imp_gs.get(c, np.zeros((1, n_feat)))[:, fi].max()
            for c in range(len(cluster_names))
        )
        ax.text(fi, y_top + 0.01, label,
                ha="center", va="bottom", fontsize=9,
                fontweight="bold" if label != "ns" else "normal")

    ax.set_xticks(positions_per_feat)
    ax.set_xticklabels(features, fontsize=9)
    ax.set_ylabel("Mean |Attribution| (GS-SHAP)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=colors[i], alpha=0.7, label=cluster_names[i])
                  for i in range(len(cluster_names))]
    ax.legend(handles=legend_els, fontsize=8, loc="upper right")

    # ── Right: Temporal Gini boxplots  [FIX-3] ────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Temporal attribution concentration  [FIXED]")

    for ci, cname in enumerate(cluster_names):
        gini_mat = gini_by_cluster.get(ci, np.zeros((1, n_feat)))  # (n, D)
        for fi in range(n_feat):
            col_data = gini_mat[:, fi]
            ax2.boxplot(
                col_data,
                positions=[positions_per_feat[fi] + offsets[ci]],
                widths=width * 0.85,
                patch_artist=True,
                medianprops=dict(color="orange", linewidth=1.5),
                boxprops=dict(facecolor=colors[ci], alpha=0.7),
                whiskerprops=dict(color=colors[ci]),
                capprops=dict(color=colors[ci]),
                flierprops=dict(marker="o", markersize=2,
                                markerfacecolor=colors[ci], alpha=0.4),
                showfliers=True,
            )

    ax2.set_xticks(positions_per_feat)
    ax2.set_xticklabels(features, fontsize=9)
    ax2.set_ylabel("Temporal Gini index  [0=uniform, 1=concentrated]")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(handles=legend_els, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Figure 9 (fixed) saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SADAF: multi-method attribution [FIXED v2]")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--target_n", type=int, default=800)
    parser.add_argument("--out_dir",  type=str, default="figures/")
    args = parser.parse_args()

    import os; os.makedirs(args.out_dir, exist_ok=True)

    df, df_paid, df_roas = load_and_preprocess(args.data_path)

    # [FIX-2] unpack group_ids
    X_reg, Y_reg, gids_reg = build_sequences(
        df_roas, "log_ROAS", FEATURES, seq_len=4)

    # [FIX-2] group-aware split
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = group_time_split(
        X_reg, Y_reg, gids_reg)
    if len(Xva) < 10 or len(Xte) < 10:
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(
            X_reg, Y_reg, 0.60, 0.80)

    ref_gru = GRUForecaster(D_IN)
    Xtr_n0, Xva_n0, Xte_n0, _ = normalize_X(Xtr, Xva, Xte)
    bs0 = min(32, len(Xtr_n0))
    ref_gru, _ = train_model(
        ref_gru,
        DataLoader(SeqDataset(Xtr_n0, Ytr), batch_size=bs0, shuffle=True),
        DataLoader(SeqDataset(Xva_n0, Yva), batch_size=bs0),
        epochs=50, patience=8)

    # PATCH: ref_lstm= → ref_model= (matches augment_pipeline signature)
    X_aug, Y_aug = augment_pipeline(
        Xtr.astype(np.float32), Ytr.astype(np.float32),
        target_n=max(args.target_n, len(Xtr) * 5), ref_model=ref_gru)

    sc = MinMaxScaler()
    N_a, T_a, D_a = X_aug.shape
    X_aug_n = sc.fit_transform(X_aug.reshape(-1, D_a)).reshape(X_aug.shape)
    X_te_n  = np.clip(sc.transform(Xte.reshape(-1, D_a)).reshape(Xte.shape), 0, 1)
    X_va_n  = np.clip(sc.transform(Xva.reshape(-1, D_a)).reshape(Xva.shape), 0, 1)

    bs   = min(128, len(X_aug_n))
    tr_l = DataLoader(SeqDataset(X_aug_n, Y_aug), batch_size=bs, shuffle=True)
    va_l = DataLoader(SeqDataset(X_va_n,  Yva),   batch_size=bs)

    print("\n══ H5: Multi-Method Attribution Comparison [FIXED] ══")
    explain_model = GRUForecaster(D_IN).to(DEVICE)
    # [FIX-1] real_val_loader for early stopping
    explain_model, _ = train_model(
        explain_model, tr_l, va_l,
        epochs=100, patience=12,
        real_val_loader=va_l)

    # Cluster on test set
    X_te_flat = X_te_n.reshape(len(X_te_n), -1)
    sc_cl     = StandardScaler()
    X_te_sc   = sc_cl.fit_transform(X_te_flat)
    km_cl     = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
    te_labels = km_cl.fit_predict(X_te_sc)
    for c in range(3):
        print(f"  Cluster {c} ({CLUSTER_NAMES[c]}): n={(te_labels==c).sum()}")

    # ── [FIX-3] GS-SHAP: collect both attribution AND Gini ────────────────
    print("\n  [1/4] GS-SHAP (HSIC grouping + Shapley) [FIX-3] ...")
    explainer = load_gsshap_explainer(explain_model, X_aug_n)

    feat_imp_gs          = {c: [] for c in range(3)}   # list of (D,) arrays
    cell_maps_by_cluster = {c: [] for c in range(3)}   # list of (T,D) arrays

    for c in range(3):
        cand      = np.where(te_labels == c)[0]
        dists     = np.linalg.norm(X_te_sc[cand] - km_cl.cluster_centers_[c], axis=1)
        sorted_ix = cand[np.argsort(dists)]
        for i, idx in enumerate(sorted_ix[:N_EXPLAIN]):
            try:
                # [FIX-3] use explain_with_gini()
                _, _, cm, gini = explainer.explain_with_gini(
                    X_te_n[idx], seed=c * 100 + i)
                feat_imp_gs[c].append(np.abs(cm).mean(axis=0))   # (D,)
                cell_maps_by_cluster[c].append(cm)                # (T,D)
            except Exception as e:
                print(f"    ⚠ GS-SHAP skipped idx={idx}: {e}")

    # Convert lists → arrays
    for c in range(3):
        feat_imp_gs[c] = (np.array(feat_imp_gs[c])
                          if feat_imp_gs[c]
                          else np.zeros((1, D_IN)))

    # [FIX-3] Compute Gini with abs-value fix
    gini_by_cluster = compute_cluster_gini(cell_maps_by_cluster)

    # ── Gini summary ───────────────────────────────────────────────────────
    print("\n  ── Temporal Gini by cluster (mean ± std) ──────────────")
    for c in range(3):
        gmat = gini_by_cluster[c]   # (n_samples, D)
        row  = "  ".join(
            f"{FEATURES[d]}={gmat[:,d].mean():.3f}±{gmat[:,d].std():.3f}"
            for d in range(D_IN)
        )
        print(f"  {CLUSTER_NAMES[c]}: {row}")

    # ── Integrated Gradients ───────────────────────────────────────────────
    print("\n  [2/4] Integrated Gradients ...")
    feat_imp_ig = {c: [] for c in range(3)}
    baseline    = X_aug_n.mean(axis=0)
    for c in range(3):
        cand      = np.where(te_labels == c)[0]
        dists     = np.linalg.norm(X_te_sc[cand] - km_cl.cluster_centers_[c], axis=1)
        sorted_ix = cand[np.argsort(dists)]
        for idx in sorted_ix[:N_EXPLAIN]:
            try:
                attr = integrated_gradients(
                    explain_model, X_te_n[idx],
                    baseline=baseline, n_steps=50, device=DEVICE)
                feat_imp_ig[c].append(np.abs(attr).mean(axis=0))
            except Exception:
                pass
        feat_imp_ig[c] = (np.array(feat_imp_ig[c])
                          if feat_imp_ig[c]
                          else np.zeros((1, D_IN)))

    # ── Permutation SHAP ──────────────────────────────────────────────────
    print("\n  [3/4] Permutation SHAP ...")
    feat_imp_perm = {c: [] for c in range(3)}
    for c in range(3):
        cand      = np.where(te_labels == c)[0]
        dists     = np.linalg.norm(X_te_sc[cand] - km_cl.cluster_centers_[c], axis=1)
        sorted_ix = cand[np.argsort(dists)]
        for idx in sorted_ix[:N_EXPLAIN]:
            try:
                feat_imp_perm[c].append(
                    permutation_shap(explain_model, X_te_n[idx],
                                     X_aug_n, n_permutations=50, device=DEVICE))
            except Exception:
                pass
        feat_imp_perm[c] = (np.array(feat_imp_perm[c])
                             if feat_imp_perm[c]
                             else np.zeros((1, D_IN)))

    # ── Attention-based ───────────────────────────────────────────────────
    print("\n  [4/4] Attention-based Attribution ...")
    attn_model = LSTMWithAttention(D_IN).to(DEVICE)
    attn_model, _ = train_model(
        attn_model, tr_l, va_l,
        epochs=100, patience=12,
        real_val_loader=va_l)   # [FIX-1]
    feat_imp_attn = {c: [] for c in range(3)}
    attn_model.eval()
    for c in range(3):
        cand      = np.where(te_labels == c)[0]
        dists     = np.linalg.norm(X_te_sc[cand] - km_cl.cluster_centers_[c], axis=1)
        sorted_ix = cand[np.argsort(dists)]
        for idx in sorted_ix[:N_EXPLAIN]:
            X_t = torch.FloatTensor(X_te_n[idx]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, w = attn_model(X_t, return_attn=True)
            w_np = w.squeeze(0).cpu().numpy()
            feat_imp_attn[c].append(
                (w_np[:, None] * np.abs(X_te_n[idx])).mean(axis=0))
        feat_imp_attn[c] = (np.array(feat_imp_attn[c])
                             if feat_imp_attn[c]
                             else np.zeros((1, D_IN)))

    # ── Spearman agreement ────────────────────────────────────────────────
    print("\n  ── Method Agreement: Spearman Rank Correlation ────")
    method_dicts = [feat_imp_gs, feat_imp_ig, feat_imp_perm, feat_imp_attn]
    for c in range(3):
        row_means = [md.get(c, np.zeros((1, D_IN))).mean(axis=0)
                     for md in method_dicts]
        corr_mat  = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                r, _ = stats.spearmanr(row_means[i], row_means[j])
                corr_mat[i, j] = r
        avg = corr_mat[np.triu_indices(4, k=1)].mean()
        print(f"  {CLUSTER_NAMES[c]:<20} avg Spearman ρ = {avg:.3f}")

    # ── Kruskal-Wallis ────────────────────────────────────────────────────
    print("\n  ── Kruskal-Wallis (GS-SHAP, primary) ──────────────")
    sig_labels = {}
    sig_feats  = []
    for fi, feat in enumerate(FEATURES):
        groups = [feat_imp_gs[c][:, fi] for c in range(3)]
        if any(len(g) == 0 for g in groups):
            sig_labels[feat] = "ns"
            continue
        stat, p = kruskal(*groups)
        if p < 0.001:
            label = "***"
        elif p < 0.01:
            label = "**"
        elif p < 0.05:
            label = "*"
        else:
            label = "ns"
        sig_labels[feat] = label
        if label != "ns":
            sig_feats.append(feat)
        print(f"  {feat:<18} p={p:.4e} {label}")

    h5 = len(sig_feats) >= 3
    print(f"\n  H5: {'SUPPORTED ✓' if h5 else 'PARTIAL ⚬'}  "
          f"({len(sig_feats)}/{len(FEATURES)} significant)")

    # ── [FIX-3] Generate corrected Figure 9 ──────────────────────────────
    out_path = f"{args.out_dir}/fig_09_gsshap_importance_fixed.png"
    plot_figure9(
        feat_imp_gs, gini_by_cluster,
        FEATURES, CLUSTER_NAMES, sig_labels,
        out_path=out_path)

    print("\n✅ Multi-method attribution (H5) [FIXED v2] complete.")


if __name__ == "__main__":
    main()
