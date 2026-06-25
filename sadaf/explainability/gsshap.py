"""
sadaf/explainability/gsshap.py  [FIXED v3]
-------------------------------------------
Changes vs. v2
--------------
FIX-3   (carried over from v2, unchanged):
  - temporal_gini() computes Gini on |cell_map|, not signed cell_map,
    preventing positive/negative attribution cancellation.

FIX-4a  (NEW — the dominant root cause of Gini ≈ 0.98–0.99 clustering
         in Figure 9, found by unit-testing temporal_gini() in
         isolation):
  - The Gini formula itself was implemented incorrectly. The v2 code
        g = (n + 1 - 2 * cumx.sum() / (total * n)) / n
    divides the *sum of the cumulative sums* by (total * n), but the
    standard Gini coefficient requires dividing the sum of *rank-
    weighted* values by (n * total):
        G = (2 * sum(i * x_i for i in 1..n)) / (n * total) - (n+1)/n
    The v2 formula is dimensionally different and does not reduce to
    0 for a perfectly uniform input. Verified by unit test: a fully
    uniform attribution vector [0.5, 0.5, 0.5, 0.5] — which should
    give Gini = 0 (no concentration at all) — gave Gini = 0.9375
    under the old formula. Some inputs even produced values above 1.0
    (e.g. 1.10), which were then silently clipped to 1.0, masking the
    bug. This alone explains why nearly every cluster/feature in the
    original run reported Gini in the 0.97–1.00 band regardless of
    the actual attribution pattern.
  - Replaced with the standard rank-weighted Gini formula (see
    temporal_gini() below). Re-verified: uniform input now correctly
    gives 0.0, and a mild 40/25/20/15 skew now gives 0.20 instead of
    0.9875.

FIX-4b  (NEW — secondary contributor: segmentation resolution):
  - Independently of FIX-4a, the time-segmentation resolution was
    also too coarse. With T=4 and max_segments=4, the old formula
        seg_len = max(min_seg_len, T // max_segments)
                = max(2, 4 // 4) = max(2, 1) = 2
    collapsed the 4 time steps into only 2 segments, so each player
    covered 2 adjacent time steps with one shared Shapley value,
    halving the temporal resolution available to any Gini-like
    concentration measure.
  - min_seg_len default changed 2 → 1, and seg_len is now computed as
    ceil(T / max_segments) instead of floor, so T=4 → 4 segments of
    length 1 (one per time step) instead of 2 segments of length 2.
  - This is a behavior change: GSSHAP now produces up to 2x more
    players (e.g. 4 → 8 for T=4, K=2 groups). Runtime scales linearly
    with n_players, so num_permutations may need to be reduced for
    very long sequences; for T<=8 this is not a practical concern.

CAVEAT (documented, not "fixed" — inherent to SEQ_LEN=4):
  - Even with both fixes applied, Gini computed over only T=4 points
    has limited resolution: there are only 4 possible "ranks" to
    spread mass across, so the achievable Gini range is coarser than
    it would be at, e.g., T=20. Cross-checking Figure 7's SEQ_LEN=6
    sequences (or computing Gini on a longer SEQ_LEN run) is
    recommended before treating any specific Gini value as precise
    rather than directionally indicative.

FIX-5   (NEW — feature-level Gini was misleading when HSIC groups
         feature multiple raw features together):
  - Added group_feature_map() and a `level` argument to
    compute_cluster_gini()/temporal_gini() so callers can request
    results keyed by HSIC group instead of raw feature. Features
    inside the same HSIC group share an identical cell_map slice by
    construction (GS-SHAP is a *group*-Shapley method), so reporting
    7 individually-named features that are actually 2 group-level
    numbers is misleading. Per-feature output is still available for
    backward compatibility but is now explicitly documented as
    "inherited from group" wherever a feature belongs to a group of
    size > 1.

All HSIC, player, and Shapley logic is otherwise unchanged from v2.
"""

from __future__ import annotations

import math
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
# [FIX-3 / FIX-4] Temporal Gini concentration index
# ─────────────────────────────────────────────────────────────────────────────

def temporal_gini(cell_map: np.ndarray) -> np.ndarray:
    """
    Compute the temporal Gini concentration index per feature column.

    [FIX-3] Uses |cell_map| so signed attributions don't cancel.
    [FIX-4a] Uses the *correct* rank-weighted Gini formula:

        G = (2 * sum_{i=1}^{n} i * x_(i)) / (n * sum(x)) - (n + 1) / n

    where x_(i) is the i-th smallest value (1-indexed). This is the
    standard Gini coefficient (equivalent to the Lorenz-curve-area
    definition). The v2 implementation divided by (total * n) instead
    of the rank-weighted sum, which does not reduce to 0 for a
    uniform input — verified by unit test (uniform [0.5,0.5,0.5,0.5]
    incorrectly gave 0.9375 under the old formula; the corrected
    formula gives exactly 0.0). See module docstring FIX-4a.

    [FIX-4b] Meaningful only when cell_map has fine enough time
            resolution (see GSSHAP segmentation fix) — with too few
            distinct time segments, Gini has fewer achievable values
            and is coarser, though no longer biased toward 1.

    NOTE: If `cell_map` was produced by a *group*-Shapley method
    (GS-SHAP), all raw feature columns that belong to the same HSIC
    group will be numerically identical by construction (see
    GSSHAP.explain()). Gini computed on those columns will therefore
    also be identical. This is expected, not a bug — see
    group_feature_map() / compute_cluster_gini(level="group") to get
    one number per HSIC group instead of D numbers that are partly
    duplicates of each other.

    Parameters
    ----------
    cell_map : np.ndarray, shape (T, D)
        Signed (T, D) attribution grid returned by GSSHAP.explain().

    Returns
    -------
    gini : np.ndarray, shape (D,)
        Temporal Gini index per feature. Values are in [0, 1].
        0 = attribution spread perfectly uniformly across all time
        steps; 1 = attribution concentrated entirely in a single step.
    """
    abs_map = np.abs(cell_map)            # (T, D)  — FIX-3
    T, D    = abs_map.shape
    gini    = np.zeros(D)

    for d in range(D):
        x = np.sort(abs_map[:, d])        # ascending sort, x_(1) <= ... <= x_(n)
        total = x.sum()
        if total < 1e-12:                 # zero attribution → Gini = 0
            gini[d] = 0.0
            continue
        n = len(x)
        rank_weighted = np.sum((np.arange(1, n + 1)) * x)   # sum_i i * x_(i)
        g = (2.0 * rank_weighted) / (n * total) - (n + 1) / n
        gini[d] = float(np.clip(g, 0.0, 1.0))

    return gini


def group_feature_map(players: list[dict]) -> dict[int, list[int]]:
    """
    [FIX-5] Recover {group_id: [feature_indices]} from a GSSHAP
    players list, so callers can tell which raw features are tied
    together (and will therefore have identical cell_map / Gini
    values) vs. which are independent.

    Parameters
    ----------
    players : list[dict]
        GSSHAP.players (or any players list with 'group_id' and
        'var_indices' keys).

    Returns
    -------
    {group_id: sorted list of feature indices in that group}
    """
    out: dict[int, list[int]] = {}
    for p in players:
        gid = p["group_id"]
        out.setdefault(gid, set()).update(p["var_indices"])
    return {gid: sorted(idxs) for gid, idxs in out.items()}


def group_temporal_gini(
    cell_map: np.ndarray,
    players: list[dict],
) -> dict[int, float]:
    """
    [FIX-5] Compute one Gini value per HSIC *group* rather than per
    raw feature. Since all features within a group share identical
    cell_map columns, this picks one representative column per group
    instead of reporting the same number D/|groups| times.

    Returns
    -------
    {group_id: gini_value}
    """
    gfm = group_feature_map(players)
    full_gini = temporal_gini(cell_map)
    return {gid: float(full_gini[idxs[0]]) for gid, idxs in gfm.items()}


def compute_cluster_gini(
    cell_maps_by_cluster: dict[int, list[np.ndarray]],
    players: list[dict] | None = None,
    level: str = "feature",
) -> dict[int, np.ndarray]:
    """
    Batch helper: compute per-sample temporal Gini for each cluster.

    Parameters
    ----------
    cell_maps_by_cluster : dict
        {cluster_id: [cell_map_1, cell_map_2, ...]}, each (T, D).
    players : list[dict] or None
        Required when level="group". GSSHAP.players (group structure
        is assumed identical across all cell_maps passed in).
    level : {"feature", "group"}
        "feature" (default, backward compatible): returns (n, D)
            arrays, one column per raw feature. Columns belonging to
            the same HSIC group will be numerically identical — see
            temporal_gini() docstring.
        "group": [FIX-5] returns (n, K) arrays, one column per HSIC
            group (K = number of HSIC groups), with no duplicated
            columns. Use this for Figure 9 reporting to avoid
            presenting the same number multiple times under different
            feature names.

    Returns
    -------
    gini_by_cluster : dict
        {cluster_id: np.ndarray of shape (n_samples, D or K)}
    """
    if level not in ("feature", "group"):
        raise ValueError("level must be 'feature' or 'group'")
    if level == "group" and players is None:
        raise ValueError("players must be provided when level='group'")

    gini_by_cluster: dict[int, np.ndarray] = {}

    if level == "feature":
        for c, cms in cell_maps_by_cluster.items():
            if not cms:
                gini_by_cluster[c] = np.zeros((1, 7))
                continue
            gini_by_cluster[c] = np.array([temporal_gini(cm) for cm in cms])
        return gini_by_cluster

    # level == "group"
    gfm = group_feature_map(players)
    group_ids = sorted(gfm.keys())
    for c, cms in cell_maps_by_cluster.items():
        if not cms:
            gini_by_cluster[c] = np.zeros((1, len(group_ids)))
            continue
        rows = []
        for cm in cms:
            g = group_temporal_gini(cm, players)
            rows.append([g[gid] for gid in group_ids])
        gini_by_cluster[c] = np.array(rows)
    return gini_by_cluster


# ─────────────────────────────────────────────────────────────────────────────
# Main GSSHAP class
# ─────────────────────────────────────────────────────────────────────────────

class GSSHAP:
    """
    Group-SHAP with HSIC-based feature grouping for temporal sequences.

    [FIX-4b] Time segmentation now defaults to one segment per time
    step (min_seg_len=1, seg_len computed via ceil instead of floor),
    so cell_map carries full temporal resolution instead of being
    coarsened into 2 wide blocks. This is a secondary contributor to
    the Gini clustering artefact described at the top of this file;
    see FIX-4a for the dominant cause (an incorrect Gini formula).
    """

    def __init__(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        task: str = "reg",
        device: torch.device | None = None,
        hsic_max_samples: int = 2000,
        min_seg_len: int = 1,          # [FIX-4] was 2
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

        # [FIX-4b] ceil-based segment length so T=4, max_segments=4 → 4
        # segments of length 1, instead of floor-based 2 segments of
        # length 2. Falls back to min_seg_len only if that would make
        # segments *larger* than requested (i.e. min_seg_len still
        # acts as a floor on resolution, not a forced coarsening).
        seg_len = max(min_seg_len, math.ceil(T / max_segments))
        self.seg_len = seg_len
        n_time_segments = math.ceil(T / seg_len)

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
        print(
            f"  [Segmentation] seg_len={seg_len}, "
            f"time_segments={n_time_segments}, "
            f"n_players={self.n_players} "
            f"(K={K} groups × {n_time_segments} time segments)"
        )
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
        Returns (phi, players, cell_map).
        cell_map shape: (T, D), signed attributions.

        NOTE: All raw feature indices that belong to the same HSIC
        group (self.feature_groups) receive an identical cell_map
        slice in the time dimension by construction — GS-SHAP assigns
        one Shapley value per (group, time-segment) player and
        broadcasts it across every feature in that group. Use
        group_feature_map(players) to see which features share values.
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

    # ── Convenience wrapper with Gini ──────────────────────────────────────
    def explain_with_gini(
        self,
        x: np.ndarray,
        seed: int = 0,
        level: str = "feature",
    ) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray | dict]:
        """
        Extended explain() that also returns temporal Gini.

        Parameters
        ----------
        level : {"feature", "group"}
            "feature": gini is np.ndarray (D,) — see temporal_gini()
                notes on duplicated values within an HSIC group.
            "group": [FIX-5] gini is dict {group_id: float}, one
                non-duplicated value per HSIC group.

        Returns
        -------
        phi      : np.ndarray (n_players,)   — Shapley values per player
        players  : list[dict]                — player metadata
        cell_map : np.ndarray (T, D)         — signed attribution grid
        gini     : np.ndarray (D,) or dict[int, float], depending on `level`
        """
        phi, players, cell_map = self.explain(x, seed=seed)
        if level == "feature":
            gini = temporal_gini(cell_map)
        elif level == "group":
            gini = group_temporal_gini(cell_map, players)
        else:
            raise ValueError("level must be 'feature' or 'group'")
        return phi, players, cell_map, gini
