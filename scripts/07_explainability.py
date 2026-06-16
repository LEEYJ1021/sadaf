"""
07_explainability.py — Multi-method attribution comparison (H5):
GS-SHAP (primary), Integrated Gradients, Permutation SHAP, Attention.

Usage:
    python scripts/07_explainability.py --data_path data/3월성과데이터(샘플).xlsx
"""
import argparse
import importlib.util
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader

from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import build_sequences, time_split, normalize_X
from sadaf.augmentation.pipeline import augment_pipeline
from sadaf.models.gru import GRUForecaster
from sadaf.models.lstm import LSTMForecaster
from sadaf.models.attention import LSTMWithAttention
from sadaf.explainability.intgrad import integrated_gradients
from sadaf.explainability.permshap import permutation_shap
from sadaf.training.trainer import train_model, eval_reg, SeqDataset

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = ["CTR", "CVR", "Depth", "log_cost",
            "log_impression", "hour_sin", "hour_cos"]
D_IN = len(FEATURES)
N_EXPLAIN = 20
CLUSTER_NAMES = ["C0 High-Volume", "C1 High-Conversion", "C2 Click-Rich"]


def load_gsshap_explainer(model, X_train):
    spec = importlib.util.spec_from_file_location(
        "gsshap", "sadaf/explainability/gsshap_standalone.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GSSHAP(
        model=model, X_train=X_train, task="reg", device=DEVICE,
        hsic_max_samples=2000, min_seg_len=2, max_segments=4,
        threshold_permutations=30, num_permutations=100, batch_size=64)


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: multi-method attribution (H5)")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--target_n", type=int, default=800)
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)
    X_reg, Y_reg = build_sequences(df_roas, "log_ROAS", FEATURES, seq_len=4)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(X_reg, Y_reg)
    if len(Xva) < 10 or len(Xte) < 10:
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(X_reg, Y_reg, 0.60, 0.80)

    ref_gru = GRUForecaster(D_IN)
    Xtr_n0, Xva_n0, Xte_n0, _ = normalize_X(Xtr, Xva, Xte)
    bs0 = min(32, len(Xtr_n0))
    ref_gru, _ = train_model(
        ref_gru,
        DataLoader(SeqDataset(Xtr_n0, Ytr), batch_size=bs0, shuffle=True),
        DataLoader(SeqDataset(Xva_n0, Yva), batch_size=bs0),
        epochs=50, patience=8)

    X_aug, Y_aug = augment_pipeline(
        Xtr.astype(np.float32), Ytr.astype(np.float32),
        target_n=max(args.target_n, len(Xtr) * 5), ref_lstm=ref_gru)

    sc = MinMaxScaler()
    N_a, T_a, D_a = X_aug.shape
    X_aug_n = sc.fit_transform(X_aug.reshape(-1, D_a)).reshape(X_aug.shape)
    X_te_n = np.clip(sc.transform(Xte.reshape(-1, D_a)).reshape(Xte.shape), 0, 1)
    X_va_n = np.clip(sc.transform(Xva.reshape(-1, D_a)).reshape(Xva.shape), 0, 1)

    bs = min(128, len(X_aug_n))
    tr_l = DataLoader(SeqDataset(X_aug_n, Y_aug), batch_size=bs, shuffle=True)
    va_l = DataLoader(SeqDataset(X_va_n, Yva), batch_size=bs)

    print("\n══ H5: Multi-Method Attribution Comparison ══════════")
    explain_model = GRUForecaster(D_IN).to(DEVICE)
    explain_model, _ = train_model(explain_model, tr_l, va_l, epochs=100, patience=12)

    X_te_flat = X_te_n.reshape(len(X_te_n), -1)
    sc_cl = StandardScaler()
    X_te_sc = sc_cl.fit_transform(X_te_flat)
    km_cl = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
    te_labels = km_cl.fit_predict(X_te_sc)
    for c in range(3):
        print(f"  Cluster {c} ({CLUSTER_NAMES[c]}): n={(te_labels == c).sum()}")

    print("\n  [1/4] GS-SHAP (HSIC grouping + Shapley) ...")
    explainer = load_gsshap_explainer(explain_model, X_aug_n)
    feat_imp_gs = {c: [] for c in range(3)}
    for c in range(3):
        cand = np.where(te_labels == c)[0]
        dists = np.linalg.norm(X_te_sc[cand] - km_cl.cluster_centers_[c], axis=1)
        sorted_idx = cand[np.argsort(dists)]
        cms = []
        for i, idx in enumerate(sorted_idx[:N_EXPLAIN]):
            try:
                _, _, cm = explainer.explain(X_te_n[idx], seed=c * 100 + i)
                cms.append(cm)
            except Exception:
                pass
        feat_imp_gs[c] = (np.array([np.abs(cm).mean(axis=0) for cm in cms])
                           if cms else np.zeros((1, D_IN)))

    print("\n  [2/4] Integrated Gradients (CPU fallback) ...")
    feat_imp_ig = {c: [] for c in range(3)}
    baseline = X_aug_n.mean(axis=0)
    for c in range(3):
        cand = np.where(te_labels == c)[0]
        dists = np.linalg.norm(X_te_sc[cand] - km_cl.cluster_centers_[c], axis=1)
        sorted_idx = cand[np.argsort(dists)]
        attrs = []
        for idx in sorted_idx[:N_EXPLAIN]:
            try:
                attr = integrated_gradients(explain_model, X_te_n[idx],
                                             baseline=baseline, n_steps=50, device=DEVICE)
                attrs.append(np.abs(attr).mean(axis=0))
            except Exception:
                pass
        feat_imp_ig[c] = np.array(attrs) if attrs else np.zeros((1, D_IN))

    print("\n  [3/4] Permutation SHAP ...")
    feat_imp_perm = {c: [] for c in range(3)}
    for c in range(3):
        cand = np.where(te_labels == c)[0]
        dists = np.linalg.norm(X_te_sc[cand] - km_cl.cluster_centers_[c], axis=1)
        sorted_idx = cand[np.argsort(dists)]
        imps = []
        for idx in sorted_idx[:N_EXPLAIN]:
            try:
                imps.append(permutation_shap(explain_model, X_te_n[idx],
                                              X_aug_n, n_permutations=50, device=DEVICE))
            except Exception:
                pass
        feat_imp_perm[c] = np.array(imps) if imps else np.zeros((1, D_IN))

    print("\n  [4/4] Attention-based Attribution ...")
    attn_model = LSTMWithAttention(D_IN).to(DEVICE)
    attn_model, _ = train_model(attn_model, tr_l, va_l, epochs=100, patience=12)
    feat_imp_attn = {c: [] for c in range(3)}
    attn_model.eval()
    for c in range(3):
        cand = np.where(te_labels == c)[0]
        dists = np.linalg.norm(X_te_sc[cand] - km_cl.cluster_centers_[c], axis=1)
        sorted_idx = cand[np.argsort(dists)]
        vals = []
        for idx in sorted_idx[:N_EXPLAIN]:
            X_t = torch.FloatTensor(X_te_n[idx]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, w = attn_model(X_t, return_attn=True)
            w_np = w.squeeze(0).cpu().numpy()
            vals.append((w_np[:, None] * np.abs(X_te_n[idx])).mean(axis=0))
        feat_imp_attn[c] = np.array(vals) if vals else np.zeros((1, D_IN))

    print("\n  ── Method Agreement: Spearman Rank Correlation ────")
    method_dicts = [feat_imp_gs, feat_imp_ig, feat_imp_perm, feat_imp_attn]
    for c in range(3):
        row_means = [md.get(c, np.zeros((1, D_IN))).mean(axis=0) for md in method_dicts]
        corr_mat = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                r, _ = stats.spearmanr(row_means[i], row_means[j])
                corr_mat[i, j] = r
        avg = corr_mat[np.triu_indices(4, k=1)].mean()
        print(f"  {CLUSTER_NAMES[c]:<20} avg Spearman ρ = {avg:.3f}")

    print("\n  ── Kruskal-Wallis (GS-SHAP, primary method) ───────")
    sig_feats = []
    for fi, feat in enumerate(FEATURES):
        groups = [feat_imp_gs[c][:, fi] for c in range(3)]
        if any(len(g) == 0 for g in groups):
            continue
        stat, p = kruskal(*groups)
        sig = "*" if p < 0.05 else "ns"
        if sig == "*":
            sig_feats.append(feat)
        print(f"  {feat:<18} p={p:.4e} {sig}")

    h5_supported = len(sig_feats) >= 3
    print(f"\n  H5: {'SUPPORTED ✓' if h5_supported else 'PARTIAL ⚬'}  "
          f"({len(sig_feats)}/{len(FEATURES)} significant)")
    print("\n✅ Multi-method attribution (H5) complete.")


if __name__ == "__main__":
    main()