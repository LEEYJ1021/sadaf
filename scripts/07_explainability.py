"""
scripts/07_explainability.py  [FIXED v3]
-----------------------------------------
Changes vs. v2
--------------
Carries the FIX-2 (group-aware split) and FIX-3 (|cell_map| Gini)
fixes from v2 unchanged. New in v3:

FIX-4a/4b (sadaf/explainability/gsshap.py):
  - Gini formula corrected (was structurally biased toward ~1.0
    regardless of true attribution pattern).
  - Time segmentation resolution increased (T=4 → 4 segments instead
    of 2), giving Gini more achievable values to work with.
  These are imported automatically via gsshap.py; no call-site changes
  are required here EXCEPT that this script now also requests
  group-level Gini (level="group") for Figure 9 reporting, since
  feature-level Gini duplicates values within an HSIC group (see
  gsshap.py FIX-5) and would otherwise mislead readers into thinking
  7 independent numbers were measured when only K (number of HSIC
  groups) independent numbers exist.

FIX-6 (sadaf/explainability/agreement.py):
  - cluster_agreement_report() now receives cluster_sizes so it can
    flag underpowered clusters (e.g. C2 n=4) directly in the printed
    report, and no longer silently prints "avg Spearman ρ = nan"
    without explanation when an attribution method's importance
    vector is near-constant for a given cluster.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans

from sadaf.config import DEVICE, RANDOM_SEED
from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import build_sequences, group_time_split
from sadaf.augmentation.pipeline import augment_pipeline
from sadaf.models.lstm import BayesianLSTM
from sadaf.training.trainer import train_model

# -- [FIX-7] Global seed fixation -------------------------------------------------
# KW significance (H5) and Spearman rho varied across runs because BayesianLSTM
# converged to a different local minimum each time (no seed control).
# Fixing all RNG sources makes the attribution analysis fully reproducible:
# same model weights -> same GS-SHAP/IntGrad/Perm-SHAP -> same KW & Spearman.
import random as _random
_SEED = RANDOM_SEED if 'RANDOM_SEED' in dir() else 42
_random.seed(_SEED)
import numpy as _np_seed; _np_seed.random.seed(_SEED)
import torch as _torch_seed
_torch_seed.manual_seed(_SEED)
_torch_seed.cuda.manual_seed_all(_SEED)
_torch_seed.backends.cudnn.deterministic = True
_torch_seed.backends.cudnn.benchmark     = False
del _random, _np_seed, _torch_seed
# ---------------------------------------------------------------------------------

from sadaf.explainability.gsshap import (
    GSSHAP,
    temporal_gini,
    group_feature_map,
    group_temporal_gini,
    compute_cluster_gini,
)
from sadaf.explainability.intgrad import integrated_gradients
from sadaf.explainability.permshap import permutation_shap
from sadaf.explainability.agreement import (
    spearman_agreement_matrix,
    average_agreement,
    cluster_agreement_report,
)

FEATURES = ["CTR", "CVR", "Depth", "log_cost", "log_impression", "hour_sin", "hour_cos"]
CLUSTER_NAMES = ["C0 High-Volume", "C1 High-Conversion", "C2 Click-Rich"]


def attention_attribution(model, X):
    """Placeholder hook — unchanged from v2. Returns per-sample (T, D)
    attention-style weights if the model exposes them, else None."""
    if not hasattr(model, "get_attention_weights"):
        return None
    return model.get_attention_weights(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", default=None,
                    help="Path to a saved BayesianLSTM .pt checkpoint. "
                         "If provided, skip training and load this model. "
                         "If not provided, train from scratch and save to "
                         "<out_dir>/best_bayesian_lstm.pt for reuse.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & preprocess ───────────────────────────────────────────
    # NOTE: load_and_preprocess() already calls _print_summary() internally
    # (see loader.py), printing the dataset summary using df["has_conversion"]
    # (whole-dataset rate, 11.77%). The previous version of this script
    # printed a second, incorrect summary here using df_paid["has_conversion"]
    # (paid-subset rate, 27.92%), which produced two different-looking
    # "Dataset summary" blocks with different Conversion % values in the
    # same run. Removed — loader.py's own summary is the single source of
    # truth and is already printed by the call below.
    df, df_paid, df_roas = load_and_preprocess(args.data_path)

    # ── [FIX-2] group-aware sequence split ─────────────────────────
    X, Y, group_ids = build_sequences(
        df_roas, "log_ROAS", seq_len=4, features=FEATURES)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = group_time_split(
        X, Y, group_ids, train_frac=0.70, val_frac=0.85)

    print(f"  Split sizes — train:{len(Xtr)}  val:{len(Xva)}  test:{len(Xte)}")

    # ── Train a reference BayesianLSTM for attribution ─────────────
    # [DIAGNOSTIC] A prior run showed "Augmentation: ..." printed TWICE
    # with a very different target_n (870, then ~41180) despite this
    # script calling augment_pipeline() exactly once at this line. Since
    # the cause could not be confirmed from logs alone, this call is now
    # wrapped with explicit before/after markers so that if a second
    # invocation is happening from somewhere else (e.g. a stale process,
    # a second cell still running, or an unexpected import-time side
    # effect), it will be unambiguous in the next run's logs.
    print(f"  [DIAGNOSTIC] augment_pipeline call site: 07_explainability.py "
          f"main(), target_n=870, len(Xtr)={len(Xtr)}, pid={os.getpid()}")
    X_aug, Y_aug = augment_pipeline(Xtr, Ytr, target_n=870)
    print(f"  [DIAGNOSTIC] augment_pipeline returned: "
          f"len(X_aug)={len(X_aug)} (expected ~870)")
    if len(X_aug) > 1000:
        print(
            "  [DIAGNOSTIC] ⚠ X_aug is far larger than the target_n=870 "
            "requested at this call site. This means augment_pipeline() "
            "is being invoked again somewhere else with a much larger "
            "target_n (likely AUG_TARGET_N from config.py, used with no "
            "override). Call stack at this point:"
        )
        import traceback
        traceback.print_stack()
    # The authoritative "Augmentation: ..." line is already printed by
    # augment_pipeline() internally. The duplicate print here was
    # producing a misleading second line with "+870 per method" (using
    # len(X_aug) instead of n_each), which looked like a second
    # augment_pipeline() call with a much larger target_n in the logs.

    from torch.utils.data import DataLoader, TensorDataset
    tr_l = DataLoader(
        TensorDataset(torch.FloatTensor(np.concatenate([Xtr, X_aug])),
                       torch.FloatTensor(np.concatenate([Ytr, Y_aug]))),
        batch_size=32, shuffle=True)
    va_l = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(Ytr)),
        batch_size=32)
    real_va_l = DataLoader(
        TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(Yva)),
        batch_size=32)

    model = BayesianLSTM(input_dim=len(FEATURES), hidden=128, dropout=0.4)

    # -- [FIX-7] Model checkpoint save/load -----------------------------------
    # If <out_dir>/best_bayesian_lstm.pt exists, load it instead of retraining.
    # This ensures attribution analysis always uses the exact same weights,
    # eliminating the KW 5/7 <-> 0/7 non-reproducibility across runs.
    # On first run (no checkpoint), trains with fixed seeds and saves the model.
    # To force retraining, delete the .pt file before running.
    _ckpt_path = Path(args.model_path) if args.model_path else (
        out_dir / "best_bayesian_lstm.pt")
    if _ckpt_path.exists():
        print(f"  [FIX-7] Loading saved model from {_ckpt_path}")
        model.load_state_dict(torch.load(_ckpt_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        history = {"train": [], "val": [], "val_real": []}
    else:
        print(f"  [FIX-7] Training from scratch (seed={_SEED}); "
              f"will save to {_ckpt_path}")
        # NOTE: trainer.py train_model() signature:
        #   train_model(model, train_loader, val_loader, epochs, lr,
        #               patience, task, weight_decay, verbose, real_val_loader)
        model, history = train_model(
            model, tr_l, va_l, real_val_loader=real_va_l,
            epochs=60, patience=12)
        torch.save(model.state_dict(), _ckpt_path)
        print(f"  [FIX-7] Model saved to {_ckpt_path}")
    # -------------------------------------------------------------------------

    print("\n══ H5: Multi-Method Attribution Comparison [FIXED v3] ══")

    # ── Cluster ad groups (on test set, by mean feature profile) ───
    X_test_mean = Xte.mean(axis=1)  # (N_test, D) — average over time
    n_clusters = 3
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_labels = km.fit_predict(X_test_mean)

    cluster_sizes = {}
    for c in range(n_clusters):
        n_c = int((cluster_labels == c).sum())
        cluster_sizes[c] = n_c
        print(f"  Cluster {c} ({CLUSTER_NAMES[c]}): n={n_c}")

    MIN_CLUSTER_N = 10  # below this, agreement stats are flagged underpowered

    # ── [1/4] GS-SHAP ────────────────────────────────────────────────
    print("\n  [1/4] GS-SHAP (HSIC grouping + Shapley) [FIX-3/4a/4b] ...")
    # NOTE: GSSHAP defaults to device=torch.device("cpu") if not given
    # explicitly. The model was already moved to DEVICE (cuda:0 if
    # available) inside train_model(), so GSSHAP's internal _predict()
    # must use the *same* device for its input tensors, or torch raises
    # "Input and parameter tensors are not at the same device". Passing
    # device=DEVICE here keeps GSSHAP's tensors and the model's
    # parameters on the same device.
    explainer = GSSHAP(model, Xtr, task="reg", device=DEVICE)
    gfm = group_feature_map(explainer.players)
    print(f"  [Reporting] HSIC groups → raw features: {gfm}")
    print(f"  [Reporting] {len(gfm)} independent group-level Gini values "
          f"will be reported; per-feature values inside a group are "
          f"identical by construction (see gsshap.py FIX-5).")

    cell_maps_by_cluster: dict[int, list[np.ndarray]] = {c: [] for c in range(n_clusters)}
    gsshap_importance_by_cluster: dict[int, np.ndarray] = {}

    for c in range(n_clusters):
        idxs = np.where(cluster_labels == c)[0]
        phis, cms = [], []
        for i in idxs:
            phi, players, cell_map = explainer.explain(Xte[i])
            phis.append(phi)
            cms.append(cell_map)
        cell_maps_by_cluster[c] = cms
        # mean |phi| per HSIC group → broadcast to feature length for
        # cross-method comparability with IntGrad/Perm-SHAP below
        mean_abs_cell = np.mean([np.abs(cm).mean(axis=0) for cm in cms], axis=0)
        gsshap_importance_by_cluster[c] = mean_abs_cell  # (D,)

    # ── Feature-level Gini (legacy / backward compatible) ───────────
    gini_feature_by_cluster = compute_cluster_gini(
        cell_maps_by_cluster, level="feature")

    # ── [FIX-5] Group-level Gini (non-duplicated, for Figure 9) ─────
    gini_group_by_cluster = compute_cluster_gini(
        cell_maps_by_cluster, players=explainer.players, level="group")

    print("\n  ── Temporal Gini by cluster (group-level, non-duplicated) ──")
    group_ids_sorted = sorted(gfm.keys())
    for c in range(n_clusters):
        row = gini_group_by_cluster[c].mean(axis=0)
        parts = "  ".join(
            f"group{gid}{gfm[gid]}={row[k]:.3f}"
            for k, gid in enumerate(group_ids_sorted)
        )
        print(f"  {CLUSTER_NAMES[c]}: {parts}")

    print("\n  ── Temporal Gini by cluster (feature-level, for reference; "
          "values repeat within an HSIC group) ──")
    for c in range(n_clusters):
        row = gini_feature_by_cluster[c].mean(axis=0)
        row_std = gini_feature_by_cluster[c].std(axis=0)
        parts = "  ".join(
            f"{f}={row[k]:.3f}±{row_std[k]:.3f}" for k, f in enumerate(FEATURES)
        )
        print(f"  {CLUSTER_NAMES[c]}: {parts}")

    # ── [2/4] Integrated Gradients ───────────────────────────────────
    # NOTE (reviewed against actual intgrad.py / permshap.py source):
    #   - permutation_shap() is safe as-is: it resolves
    #     `device = next(model.parameters()).device` when not given, so
    #     it always matches wherever `model` currently lives.
    #   - integrated_gradients() temporarily moves `model` itself (not a
    #     copy — nn.Module.cpu()/.to() mutate and return the same object)
    #     to CPU, runs the IG loop in train() mode to keep MC-Dropout
    #     active, then moves it back to its original device and restores
    #     eval()/train() mode at the end of the function body. This is
    #     fine when the function returns normally. The risk is if an
    #     exception is raised mid-loop (e.g. a CUDA OOM from a concurrent
    #     process, a NaN, a KeyboardInterrupt): the restore lines never
    #     run, and `model` would be left on CPU with stale .grad values
    #     from the last backward() call, breaking the subsequent
    #     permutation_shap() / GSSHAP calls below with the same device
    #     mismatch fixed earlier for GSSHAP. Wrapping the call site in
    #     try/finally below guarantees model.to(DEVICE) is re-applied
    #     and stray gradients are cleared even if integrated_gradients()
    #     raises partway through a cluster's sample loop.
    print("\n  [2/4] Integrated Gradients ...")
    intgrad_importance_by_cluster = {}
    try:
        for c in range(n_clusters):
            idxs = np.where(cluster_labels == c)[0]
            ig_vals = [integrated_gradients(model, Xte[i]) for i in idxs]
            intgrad_importance_by_cluster[c] = np.mean(
                [np.abs(v).mean(axis=0) for v in ig_vals], axis=0)
    finally:
        # Defensive: integrated_gradients() should already have restored
        # model's device/mode on normal return, but re-assert it here in
        # case an exception interrupted that restore for any sample.
        model.to(DEVICE)
        model.zero_grad(set_to_none=True)
        model.eval()

    # ── [3/4] Permutation SHAP ───────────────────────────────────────
    print("\n  [3/4] Permutation SHAP ...")
    permshap_importance_by_cluster = {}
    for c in range(n_clusters):
        idxs = np.where(cluster_labels == c)[0]
        # [BUG-FIX] Previously called without `seed`, so permutation_shap()
        # used a hard-coded rng = np.random.default_rng(42) internally —
        # resetting to the same state on every call and producing identical
        # background-index sequences for every x_sample. This made the
        # importance vectors near-constant across all samples and clusters
        # (std ≈ 0), triggering "near-constant → Spearman undefined" in
        # agreement.py.  Fix: pass seed=int(idxs[j]) so each sample draws
        # a different background sequence while remaining reproducible.
        ps_vals = [
            permutation_shap(model, Xte[i], Xtr, seed=int(i))
            for i in idxs
        ]
        permshap_importance_by_cluster[c] = np.mean(
            np.abs(np.stack(ps_vals)), axis=0)  # [BUG-FIX] (n_samples,D)->mean->(D,) not scalar

    # ── [4/4] Attention-based attribution (excluded from consensus) ─
    print("\n  [4/4] Attention-based Attribution ...")
    # Attention measures temporal position, not feature importance —
    # kept out of the Spearman agreement matrix (see agreement.py docstring
    # and Figure W4). Retained here only for the separate temporal-position
    # comparison figure, not for H5 agreement statistics.

    # ── [FIX-6] Method agreement with explicit NaN diagnosis ─────────
    print("\n  ── Method Agreement: Spearman Rank Correlation ────")
    method_names = ["GS-SHAP", "IntGrad", "Perm-SHAP"]
    cluster_mean_importances = {
        c: [
            gsshap_importance_by_cluster[c],
            intgrad_importance_by_cluster[c],
            permshap_importance_by_cluster[c],
        ]
        for c in range(n_clusters)
    }
    agreement_report = cluster_agreement_report(
        cluster_mean_importances,
        method_names=method_names,
        cluster_names=CLUSTER_NAMES,
        min_cluster_n=MIN_CLUSTER_N,
        cluster_sizes=cluster_sizes,
    )

    underpowered_clusters = [
        CLUSTER_NAMES[c] for c, r in agreement_report.items() if r["underpowered"]
    ]
    if underpowered_clusters:
        print(f"\n  ⚠ NOTE: {underpowered_clusters} have n < {MIN_CLUSTER_N} "
              f"test samples. Agreement and Kruskal-Wallis statistics for "
              f"these clusters should be reported with this caveat, not as "
              f"unconditional null/positive findings.")

    # ── Kruskal-Wallis (GS-SHAP, primary) ───────────────────────────
    print("\n  ── Kruskal-Wallis (GS-SHAP, primary) ──────────────")
    kw_results = {}
    for k, feat in enumerate(FEATURES):
        groups = [
            np.array([np.abs(cm[:, k]).mean() for cm in cell_maps_by_cluster[c]])
            for c in range(n_clusters)
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            print(f"  {feat:<18} skipped (insufficient n per cluster)")
            continue
        h_stat, p_val = scipy_stats.kruskal(*groups)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        kw_results[feat] = (h_stat, p_val)
        print(f"  {feat:<18} p={p_val:.4e} {sig}")

    n_sig = sum(1 for _, p in kw_results.values() if p < 0.05)
    print(f"\n  H5: {'SUPPORTED ✓' if n_sig >= 4 else 'PARTIAL ⚬' if n_sig >= 1 else 'NULL ⚬'}"
          f"  ({n_sig}/{len(kw_results)} significant)")
    if underpowered_clusters:
        print(f"  → Caveat: result is influenced by underpowered cluster(s) "
              f"{underpowered_clusters}; see note above.")

    # ── Figure 9 (group-level Gini, non-duplicated) ──────────────────
    fig_path = out_dir / "fig_09_gsshap_importance_fixed.png"
    _plot_figure9(
        gsshap_importance_by_cluster,
        gini_group_by_cluster,
        gfm,
        group_ids_sorted,
        fig_path,
    )
    print(f"  → Figure 9 (fixed) saved to {fig_path}")

    print("\n✅ Multi-method attribution (H5) [FIXED v3] complete.")


def _plot_figure9(importance_by_cluster, gini_group_by_cluster, gfm,
                   group_ids_sorted, out_path):
    """Left: GS-SHAP importance boxplots by cluster (feature-level).
    Right: group-level temporal Gini boxplots (one box per HSIC group,
    not one per raw feature) — fixes the v2/v3 ambiguity where 7
    feature-named boxes really only carried K independent values."""
    n_clusters = len(importance_by_cluster)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    data = [importance_by_cluster[c] for c in range(n_clusters)]
    arr = np.array(data)          # expected (n_clusters, D)
    # Guard against shape mismatch: if importance vectors have fewer dims
    # than FEATURES (e.g. group-level instead of feature-level), derive
    # tick_labels from the actual column count rather than assuming D=7.
    n_cols = arr.shape[1] if arr.ndim == 2 else len(FEATURES)
    plot_labels = FEATURES if n_cols == len(FEATURES) else [f"group{i}" for i in range(n_cols)]
    # NOTE: matplotlib renamed Axes.boxplot()'s `labels` kwarg to
    # `tick_labels` (removed in this environment's matplotlib 3.10.8).
    ax.boxplot(arr, tick_labels=plot_labels)    # [BUG-FIX] arr not transposed
    ax.set_title("GS-SHAP mean |attribution| by feature")
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1]
    group_labels = [f"group{gid} {gfm[gid]}" for gid in group_ids_sorted]
    box_data = [
        [gini_group_by_cluster[c][:, k] for c in range(n_clusters)]
        for k in range(len(group_ids_sorted))
    ]
    flat_data = [arr for grp in box_data for arr in grp]
    positions = []
    pos = 1
    tick_pos = []
    for k in range(len(group_ids_sorted)):
        tick_pos.append(pos + (n_clusters - 1) / 2)
        for c in range(n_clusters):
            positions.append(pos)
            pos += 1
        pos += 1
    ax.boxplot(flat_data, positions=positions, widths=0.8)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(group_labels, rotation=20, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title("Temporal Gini by HSIC group (FIX-4a/4b corrected)")
    ax.set_ylabel("Gini (0=uniform over time, 1=concentrated)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
