"""
sadaf/explainability/agreement.py  [FIXED v2]
Spearman ρ method agreement across attribution methods (SADAF §7 / H5).

FIX-6 (NEW — root cause of "avg Spearman ρ = nan" in the H5 run log):
  scipy.stats.spearmanr returns NaN whenever either input vector has
  zero variance (a constant vector), because rank correlation is
  undefined when there is nothing to rank against. This happens here
  for two compounding reasons, both upstream of this file:
    1. [gsshap.py FIX-4] Before the segmentation fix, GS-SHAP grouped
       multiple raw features into the same HSIC group, so several
       entries of the (D,) importance vector were *identical by
       construction*. A vector with 5 of 7 entries equal to each
       other is not technically constant, but it sharply reduces the
       effective variance Spearman has to work with, and in small-n
       clusters (e.g. C2, n=4) the per-sample-averaged importance
       vector can become exactly constant after rounding/aggregation.
    2. Small cluster sizes (C2 n=4) amplify (1): with very few samples
       contributing to the mean importance vector, ties and constant
       columns are far more likely.
  This file cannot fix the upstream cause (see gsshap.py FIX-4 / 5),
  but it must no longer silently emit NaN into a printed report. The
  fix here is defensive + diagnostic:
    - spearman_agreement_matrix() now detects near-constant inputs
      *before* calling spearmanr, sets that cell to NaN explicitly
      (instead of relying on scipy's own NaN), and returns a parallel
      boolean mask so callers know which pairs were skipped and why.
    - average_agreement() now ignores NaN cells (nanmean) instead of
      propagating NaN into the printed average, and reports how many
      of the upper-triangle pairs were actually usable.
    - cluster_agreement_report() prints an explicit warning per
      cluster when one or more methods had near-constant importance,
      naming the method, instead of printing "avg Spearman ρ = nan"
      with no explanation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from itertools import combinations

# Below this standard deviation, a vector is treated as "near-constant"
# and excluded from Spearman computation rather than passed to scipy
# (which would silently return NaN with no diagnostic).
_CONST_STD_TOL = 1e-10


def _is_near_constant(x: np.ndarray) -> bool:
    return float(np.std(x)) < _CONST_STD_TOL


def spearman_agreement_matrix(
    method_importances: list[np.ndarray],
    method_names: list[str] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise Spearman ρ matrix for M attribution methods.

    [FIX-6] Near-constant importance vectors are detected explicitly
    and excluded *before* calling scipy, so a NaN in the output always
    has a known, reportable cause rather than appearing silently.

    Parameters
    ----------
    method_importances : list of np.ndarray, each shape (D,)
        Mean feature importance vector from each method.
    method_names : list[str] or None
        Optional names for diagnostic printing.
    verbose : bool
        If True, print which method(s) were near-constant.

    Returns
    -------
    corr_mat : np.ndarray, shape (M, M)
        NaN on cells involving a near-constant input.
    valid_mask : np.ndarray, shape (M, M), bool
        True where corr_mat[i, j] is a real (non-NaN-by-construction)
        Spearman value.
    """
    M = len(method_importances)
    corr_mat = np.full((M, M), np.nan)
    valid_mask = np.zeros((M, M), dtype=bool)

    const_flags = [_is_near_constant(np.asarray(v)) for v in method_importances]
    if verbose and any(const_flags):
        names = method_names or [f"method_{i}" for i in range(M)]
        flagged = [names[i] for i, f in enumerate(const_flags) if f]
        print(f"    ⚠ near-constant importance vector(s): {flagged} "
              f"(std < {_CONST_STD_TOL}) — Spearman undefined, set to NaN")

    for i in range(M):
        for j in range(M):
            if i == j:
                corr_mat[i, j] = 1.0
                valid_mask[i, j] = True
                continue
            if const_flags[i] or const_flags[j]:
                continue  # leave as NaN, not valid
            r, _ = stats.spearmanr(method_importances[i], method_importances[j])
            corr_mat[i, j] = r
            valid_mask[i, j] = not np.isnan(r)

    return corr_mat, valid_mask


def average_agreement(
    corr_mat: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[float, int, int]:
    """
    Mean of upper-triangle Spearman ρ values (excl. diagonal).

    [FIX-6] Uses nanmean over only the *valid* upper-triangle cells
    instead of a plain mean, so a single near-constant method no
    longer turns the entire average into NaN.

    Returns
    -------
    avg : float
        NaN only if zero valid pairs exist (all methods near-constant).
    n_valid : int
        Number of usable (non-NaN) pairs.
    n_total : int
        Total number of upper-triangle pairs (for reporting, e.g. "3/6").
    """
    M = corr_mat.shape[0]
    indices = np.triu_indices(M, k=1)
    vals = corr_mat[indices]
    n_total = len(vals)

    if valid_mask is not None:
        mask_vals = valid_mask[indices]
        usable = vals[mask_vals]
    else:
        usable = vals[~np.isnan(vals)]

    n_valid = len(usable)
    if n_valid == 0:
        return float("nan"), 0, n_total
    return float(np.mean(usable)), n_valid, n_total


def rank_consensus_matrix(
    method_importances: list[np.ndarray],
) -> np.ndarray:
    """
    Feature rank matrix across methods.

    Parameters
    ----------
    method_importances : list of np.ndarray, each shape (D,)

    Returns
    -------
    rank_mat : np.ndarray, shape (M, D)
        rank_mat[i, j] = rank of feature j under method i (1 = most important).
    """
    D = len(method_importances[0])
    rank_mat = np.zeros((len(method_importances), D))
    for i, imp in enumerate(method_importances):
        rank_mat[i] = D - np.argsort(np.argsort(imp))
    return rank_mat


def cluster_agreement_report(
    cluster_mean_importances: dict[int, list[np.ndarray]],
    method_names: list[str],
    cluster_names: list[str] | None = None,
    min_cluster_n: int | None = None,
    cluster_sizes: dict[int, int] | None = None,
) -> dict[int, dict]:
    """
    Compute agreement metrics across methods for each cluster.

    [FIX-6] No longer prints a bare "avg Spearman ρ = nan" with no
    explanation. Instead:
      - flags and names which method(s) had a near-constant
        importance vector in that cluster,
      - reports the average as "X/Y pairs usable",
      - optionally flags clusters below `min_cluster_n` as
        underpowered (e.g. C2 with n=4) so the printed report makes
        the sample-size caveat visible at the point of use, not only
        in a downstream write-up.

    Parameters
    ----------
    cluster_mean_importances : dict
        {cluster_id: [importance_array_method1, importance_method2, ...]}
    method_names : list[str]
        Names of attribution methods (same order as importances).
    cluster_names : list[str] or None
    min_cluster_n : int or None
        If provided together with `cluster_sizes`, clusters with
        cluster_sizes[c] < min_cluster_n are flagged as underpowered.
    cluster_sizes : dict[int, int] or None
        {cluster_id: n_samples}, used only for the underpowered flag.

    Returns
    -------
    report : dict
        {cluster_id: {'matrix': corr_mat, 'valid_mask': ..., 'avg': float,
                       'n_valid_pairs': int, 'n_total_pairs': int,
                       'label': str, 'underpowered': bool}}
    """
    report = {}
    for c, means in cluster_mean_importances.items():
        corr_mat, valid_mask = spearman_agreement_matrix(
            means, method_names=method_names, verbose=True,
        )
        avg, n_valid, n_total = average_agreement(corr_mat, valid_mask)

        if np.isnan(avg):
            level = "Undefined (no usable pairs)"
        else:
            level = (
                "High" if avg > 0.7
                else "Moderate" if avg > 0.4
                else "Low"
            )

        label = cluster_names[c] if cluster_names else f"Cluster {c}"

        underpowered = False
        n_str = ""
        if min_cluster_n is not None and cluster_sizes is not None:
            n_c = cluster_sizes.get(c)
            if n_c is not None:
                underpowered = n_c < min_cluster_n
                n_str = f", n={n_c}" + (" ⚠ underpowered" if underpowered else "")

        report[c] = dict(
            matrix=corr_mat,
            valid_mask=valid_mask,
            avg=avg,
            n_valid_pairs=n_valid,
            n_total_pairs=n_total,
            level=level,
            cluster_label=label,
            underpowered=underpowered,
        )

        avg_str = "nan" if np.isnan(avg) else f"{avg:.3f}"
        print(
            f"  {label:<22}  avg Spearman ρ = {avg_str}  "
            f"({level}, {n_valid}/{n_total} pairs usable{n_str})"
        )
    return report
