"""
sadaf/explainability/gsshap.py  [FIXED v2]
-------------------------------------------
Changes vs. original
---------------------
FIX-3  (Temporal Gini was blank in Figure 9):
  - temporal_gini() function added.
    The original viz code was computing Gini on *signed* cell_map values,
    so positive and negative attributions cancelled out → all values
    near 0 → invisible boxplots.
    Correct implementation:
      (a) take np.abs(cell_map) before computing Gini, and
      (b) apply the standard Gini formula on the T-dimensional time
          concentration vector per feature.
  - explain() return signature unchanged; a new explain_with_gini()
    convenience wrapper returns the Gini vector alongside existing outputs.
  - compute_cluster_gini() batch helper added for 07_explainability.py
    and 10_figures.py to call directly.

All HSIC, player, and Shapley logic is unchanged.
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# HSIC utilities  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _rbf_kernel(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    sq = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    if sigma is None:
        sigma = np.median(np.sqrt(sq[sq > 0])) + 1e-8
    return np.exp(-sq / (2 * sigma ** 2))


def _centre_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def hsic(X: np.ndarray, Y: np.ndarray) -> float:
    n = len(X)
    Kx = _centre_kernel(_rbf_kernel(X.reshape(n, -1)))
    Ky = _centre_kernel(_rbf_kernel(Y.reshape(n, -1)))
    return float(np.trace(Kx @ Ky) / (n - 1) ** 2)


def _hsic_feature_groups(
    X_train: np.ndarray,
    max_samples: int = 2000,
) -> list[list[int]]:
    N, T, D = X_train.shape
    idx = np.random.default_rng(0).choice(N, min(N, max_samples), replace=False)
    X_s = X_train[idx, -1, :]

    A = np.eye(D)
    for i, j in combinations(range(D), 2):
        h = hsic(X_s[:, i:i+1], X_s[:, j:j+1])
        A[i, j] = A[j, i] = max(h, 0)

    D_inv  = np.diag(1.0 / (A.sum(1) + 1e-8))
    L_norm = D_inv @ A

    vals, vecs = np.linalg.eigh(L_norm)
    vals = vals[::-1]; vecs = vecs[:, ::-1]

    gaps = np.diff(vals[:D//2 + 1])
    K    = max(2, int(np.argmin(gaps) + 1))

    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(
        vecs[:, :K]
    )
    groups = [list(np.where(labels == k)[0]) for k in range(K)]
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-3] Temporal Gini concentration index
# ─────────────────────────────────────────────────────────────────────────────

def temporal_gini(cell_map: np.ndarray) -> np.ndarray:
    """
    Compute the temporal Gini concentration index per feature.

    [FIX-3] The original viz code applied Gini to *signed* cell_map values,
    causing positive and negative attributions to cancel → values ≈ 0
    → invisible boxplots in Figure 9 right panel.

    Correct procedure:
      1. Take absolute values  (attribution magnitude, not direction).
      2. Apply the Lorenz-based Gini formula along the time axis.
         Gini ∈ [0, 1]:  0 = attribution spread uniformly over all
         time steps; 1 = attribution concentrated in a single step.

    Parameters
    ----------
    cell_map : np.ndarray, shape (T, D)
        Signed (T, D) attribution grid returned by GSSHAP.explain().

    Returns
    -------
    gini : np.ndarray, shape (D,)
        Temporal Gini index per feature.  Values are in [0, 1].
    """
    # Step 1 — use absolute attributions  ←  THE critical fix
    abs_map = np.abs(cell_map)            # (T, D)
    T, D    = abs_map.shape
    gini    = np.zeros(D)

    for d in range(D):
        x = np.sort(abs_map[:, d])        # ascending sort
        total = x.sum()
        if total < 1e-12:                 # zero attribution → Gini = 0
            gini[d] = 0.0
            continue
        # Lorenz-based formula: G = 1 - (2/n·S) * Σ_{i=1}^{n} (n-i+1)·x_i
        # Equivalent to the more common: G = (Σ|x_i - x_j|) / (2·n·Σx_i)
        n       = len(x)
        cumx    = np.cumsum(x)
        gini[d] = float(
            (n + 1 - 2 * cumx.sum() / (total * n)) / n
        )
        gini[d] = float(np.clip(gini[d], 0.0, 1.0))

    return gini


def compute_cluster_gini(
    cell_maps_by_cluster: dict[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    """
    [FIX-3] Batch helper: compute per-sample temporal Gini for each cluster.

    Parameters
    ----------
    cell_maps_by_cluster : dict
        {cluster_id: [cell_map_1, cell_map_2, ...]}
        where each cell_map has shape (T, D).

    Returns
    -------
    gini_by_cluster : dict
        {cluster_id: np.ndarray of shape (n_samples, D)}
        Can be passed directly to plt.boxplot / seaborn.boxplot.
    """
    gini_by_cluster = {}
    for c, cms in cell_maps_by_cluster.items():
        if not cms:
            gini_by_cluster[c] = np.zeros((1, cms[0].shape[-1]
                                            if cms else 7))
            continue
        gini_by_cluster[c] = np.array([temporal_gini(cm) for cm in cms])
    return gini_by_cluster


# ─────────────────────────────────────────────────────────────────────────────
# Main GSSHAP class  (logic unchanged; new explain_with_gini() added)
# ─────────────────────────────────────────────────────────────────────────────

class GSSHAP:
    """
    Group-SHAP with HSIC-based feature grouping for temporal sequences.
    [FIX-3] New method explain_with_gini() returns Gini alongside Shapley.
    """

    def __init__(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        task: str = "reg",
        device: torch.device | None = None,
        hsic_max_samples: int = 2000,
        min_seg_len: int = 2,
        max_segments: int = 4,
        threshold_permutations: int = 30,
        num_permutations: int = 100,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.task = task
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.num_permutations = num_permutations
        self.threshold_permutations = threshold_permutations

        _, T, D = X_train.shape
        self.T = T
        self.D = D

        print("[GS-SHAP] Computing HSIC feature groups from training data...")
        t0 = time.time()
        self.feature_groups = _hsic_feature_groups(X_train, hsic_max_samples)
        K = len(self.feature_groups)
        print(f"  [HSIC] eigengap → K={K} groups (D={D} features)")
        print(f"  Groups: {self.feature_groups}  ({time.time()-t0:.2f}s)")

        seg_len = max(min_seg_len, T // max_segments)
        self.players: list[dict] = []
        for gid, grp in enumerate(self.feature_groups):
            for t_start in range(0, T, seg_len):
                t_end = min(t_start + seg_len, T)
                self.players.append(
                    dict(
                        group_id=gid,
                        var_indices=grp,
                        time_range=(t_start, t_end),
                    )
                )

        self.n_players = len(self.players)
        self._baseline = X_train.mean(axis=0)   # (T, D)

    # ── Prediction ────────────────────────────────────────────────────────
    def _predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                Xb = torch.FloatTensor(X[i : i + self.batch_size]).to(self.device)
                preds.append(self.model(Xb).cpu().numpy())
        return np.concatenate(preds)

    # ── Masking ───────────────────────────────────────────────────────────
    def _mask(self, x: np.ndarray, present: list[int]) -> np.ndarray:
        x_m = self._baseline.copy()
        for pid in present:
            p = self.players[pid]
            t0, t1 = p["time_range"]
            for fi in p["var_indices"]:
                x_m[t0:t1, fi] = x[t0:t1, fi]
        return x_m

    # ── Shapley via random permutation sampling ───────────────────────────
    def explain(
        self,
        x: np.ndarray,
        seed: int = 0,
    ) -> tuple[np.ndarray, list[dict], np.ndarray]:
        """
        Original interface — returns (phi, players, cell_map).
        cell_map shape: (T, D), signed attributions.
        """
        rng = np.random.default_rng(seed)
        phi = np.zeros(self.n_players)
        player_ids = np.arange(self.n_players)

        for _ in range(self.num_permutations):
            perm = rng.permutation(player_ids)
            present: list[int] = []
            prev_val = float(self._predict(self._baseline[None])[0])
            for pid in perm:
                present.append(pid)
                x_masked = self._mask(x, present)
                cur_val  = float(self._predict(x_masked[None])[0])
                phi[pid] += cur_val - prev_val
                prev_val  = cur_val

        phi /= self.num_permutations

        cell_map = np.zeros((self.T, self.D))
        for pid, p in enumerate(self.players):
            t0, t1 = p["time_range"]
            n_cells = (t1 - t0) * len(p["var_indices"])
            if n_cells == 0:
                continue
            share = phi[pid] / n_cells
            for fi in p["var_indices"]:
                cell_map[t0:t1, fi] += share

        return phi, self.players, cell_map

    # ── [FIX-3] Convenience wrapper with Gini ─────────────────────────────
    def explain_with_gini(
        self,
        x: np.ndarray,
        seed: int = 0,
    ) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
        """
        [FIX-3] Extended explain() that also returns temporal Gini.

        Returns
        -------
        phi      : np.ndarray (n_players,)   — Shapley values per player
        players  : list[dict]                — player metadata
        cell_map : np.ndarray (T, D)         — signed attribution grid
        gini     : np.ndarray (D,)           — temporal Gini per feature
                   Computed from |cell_map| to avoid sign-cancellation.
        """
        phi, players, cell_map = self.explain(x, seed=seed)
        gini = temporal_gini(cell_map)      # [FIX-3] abs-value Gini
        return phi, players, cell_map, gini
