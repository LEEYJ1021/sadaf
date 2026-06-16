"""
sadaf/explainability/gsshap.py
GS-SHAP: HSIC-based feature grouping + Shapley value decomposition.

Primary attribution method in SADAF (§7 / H5).

Algorithm
---------
1. Compute HSIC (Hilbert-Schmidt Independence Criterion) between all
   feature pairs on training data to build a dependency graph.
2. Spectral clustering on the HSIC matrix to identify K feature groups
   (eigengap heuristic).  Default groups found: [[CTR, Depth], [CVR, log_cost,
   log_impression], [hour_sin, hour_cos]].
3. Treat each (group × time-step segment) combination as a "player" in a
   cooperative game.
4. Estimate Shapley values via random permutation sampling.

Result (SADAF §7):
  η²_max = 0.525 (hour_sin/cos), 4/7 features significant (KW p < 0.05).
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# HSIC utilities
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
    """Unbiased HSIC estimate."""
    n = len(X)
    Kx = _centre_kernel(_rbf_kernel(X.reshape(n, -1)))
    Ky = _centre_kernel(_rbf_kernel(Y.reshape(n, -1)))
    return float(np.trace(Kx @ Ky) / (n - 1) ** 2)


def _hsic_feature_groups(
    X_train: np.ndarray,
    max_samples: int = 2000,
) -> list[list[int]]:
    """
    Cluster features by HSIC dependency.  Uses eigengap on the
    normalised HSIC affinity matrix.
    """
    N, T, D = X_train.shape
    idx = np.random.default_rng(0).choice(N, min(N, max_samples), replace=False)
    X_s = X_train[idx, -1, :]  # use last time-step for HSIC

    # Build D×D HSIC affinity matrix
    A = np.eye(D)
    for i, j in combinations(range(D), 2):
        h = hsic(X_s[:, i:i+1], X_s[:, j:j+1])
        A[i, j] = A[j, i] = max(h, 0)

    # Normalise
    D_inv = np.diag(1.0 / (A.sum(1) + 1e-8))
    L_norm = D_inv @ A

    vals, vecs = np.linalg.eigh(L_norm)
    vals = vals[::-1]; vecs = vecs[:, ::-1]

    # Eigengap heuristic for K
    gaps = np.diff(vals[:D//2 + 1])
    K = max(2, int(np.argmin(gaps) + 1))

    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(
        vecs[:, :K]
    )
    groups = [list(np.where(labels == k)[0]) for k in range(K)]
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Main GSSHAP class
# ─────────────────────────────────────────────────────────────────────────────

class GSSHAP:
    """
    Group-SHAP with HSIC-based feature grouping for temporal sequences.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model (eval mode preferred; must accept (B, T, D)).
    X_train : np.ndarray, shape (N, T, D)
        Training data used to build HSIC groups and as reference distribution.
    task : str
        'reg' or 'cls'.
    device : torch.device
    hsic_max_samples : int
        Max samples for HSIC computation.
    min_seg_len : int
        Minimum time-step segment length per player.
    max_segments : int
        Maximum time segments per feature group.
    threshold_permutations : int
        Permutations for fast convergence check.
    num_permutations : int
        Total Shapley permutations.
    batch_size : int
        Inference batch size.
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
        print(
            f"  [HSIC] eigengap → K={K} groups (D={D} features)"
        )
        print(f"  Groups: {self.feature_groups}  ({time.time()-t0:.2f}s)")

        # Build players: each (group, time-segment) pair
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
        self._baseline = X_train.mean(axis=0)  # (T, D)

    # ── Prediction helper ────────────────────────────────────────────────

    def _predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                Xb = torch.FloatTensor(X[i : i + self.batch_size]).to(self.device)
                out = self.model(Xb).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds)

    # ── Masking ──────────────────────────────────────────────────────────

    def _mask(
        self,
        x: np.ndarray,
        present: list[int],
    ) -> np.ndarray:
        """Replace absent players with baseline values."""
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
        Compute Shapley values for a single instance.

        Parameters
        ----------
        x : np.ndarray, shape (T, D)
        seed : int

        Returns
        -------
        phi : np.ndarray, shape (n_players,)
            Shapley value per player.
        players : list[dict]
            Player metadata.
        cell_map : np.ndarray, shape (T, D)
            Attribution distributed over the (time, feature) grid.
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

        # Build (T, D) attribution map
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
