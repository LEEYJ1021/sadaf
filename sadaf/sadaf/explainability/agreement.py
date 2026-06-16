"""
sadaf/explainability/agreement.py
Spearman ρ method agreement across attribution methods (SADAF §7 / H5).

SADAF result:
  3-method consensus (GS-SHAP / IntGrad / Perm-SHAP):
    C0 High-Volume      avg ρ = 0.719  (High)
    C1 High-Conversion  avg ρ = 0.624  (Moderate)
    C2 Click-Rich       avg ρ = 0.699  (Moderate)

  Attention is EXCLUDED from consensus (W4 reframing):
    Attention measures temporal position (which time-step); gradient-based
    methods measure feature-level importance.  These are complementary axes.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from itertools import combinations


def spearman_agreement_matrix(
    method_importances: list[np.ndarray],
) -> np.ndarray:
    """
    Compute pairwise Spearman ρ matrix for M attribution methods.

    Parameters
    ----------
    method_importances : list of np.ndarray, each shape (D,)
        Mean feature importance vector from each method.

    Returns
    -------
    corr_mat : np.ndarray, shape (M, M)
    """
    M = len(method_importances)
    corr_mat = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            r, _ = stats.spearmanr(method_importances[i], method_importances[j])
            corr_mat[i, j] = r
    return corr_mat


def average_agreement(corr_mat: np.ndarray) -> float:
    """Mean of upper-triangle Spearman ρ values (excl. diagonal)."""
    M = corr_mat.shape[0]
    indices = np.triu_indices(M, k=1)
    return float(corr_mat[indices].mean())


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
) -> dict[int, dict]:
    """
    Compute agreement metrics across methods for each cluster.

    Parameters
    ----------
    cluster_mean_importances : dict
        {cluster_id: [importance_array_method1, importance_method2, ...]}
    method_names : list[str]
        Names of attribution methods (same order as importances).
    cluster_names : list[str] or None

    Returns
    -------
    report : dict
        {cluster_id: {'matrix': corr_mat, 'avg': float, 'label': str}}
    """
    report = {}
    for c, means in cluster_mean_importances.items():
        corr_mat = spearman_agreement_matrix(means)
        avg = average_agreement(corr_mat)
        level = (
            "High" if avg > 0.7
            else "Moderate" if avg > 0.4
            else "Low"
        )
        label = cluster_names[c] if cluster_names else f"Cluster {c}"
        report[c] = dict(
            matrix=corr_mat,
            avg=avg,
            level=level,
            cluster_label=label,
        )
        print(
            f"  {label:<22}  avg Spearman ρ = {avg:.3f}  ({level} agreement)"
        )
    return report
