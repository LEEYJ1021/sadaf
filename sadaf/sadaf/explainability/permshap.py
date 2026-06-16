"""
sadaf/explainability/permshap.py
Permutation-based SHAP for feature importance in SADAF (§7 / H5).

Method:
  For each feature dimension d, replace the input's values at dimension d
  with a random background sample and measure the output change.
  Repeat n_permutations times and average |Δf| per feature.

This is a model-agnostic, gradient-free alternative to GS-SHAP that
does not require access to model internals.  It measures *marginal*
feature importance averaged over time — complementary to GS-SHAP's
group-and-temporal decomposition.

SADAF W4 note:
  Perm-SHAP and GS-SHAP show moderate-to-high agreement (Spearman ρ ≥ 0.62)
  in the three gradient-based method consensus (excl. Attention).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def permutation_shap(
    model: nn.Module,
    x_sample: np.ndarray,
    X_background: np.ndarray,
    n_permutations: int = 100,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Compute feature importance for a single input via permutation SHAP.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model accepting (1, T, D) input.
    x_sample : np.ndarray, shape (T, D)
        Input to explain.
    X_background : np.ndarray, shape (N, T, D)
        Reference distribution for feature replacement.
    n_permutations : int
        Number of random background replacements per feature.
    device : torch.device or None

    Returns
    -------
    importances : np.ndarray, shape (D,)
        Mean |Δoutput| per feature dimension.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    T, D = x_sample.shape
    rng  = np.random.default_rng(42)

    # Baseline prediction
    x_t = torch.FloatTensor(x_sample).unsqueeze(0).to(device)
    with torch.no_grad():
        base_out = float(model(x_t).item())

    importances = np.zeros(D)
    for fi in range(D):
        diffs = []
        for _ in range(n_permutations):
            x_perm = x_sample.copy()
            bg_idx = rng.integers(0, len(X_background))
            x_perm[:, fi] = X_background[bg_idx, :, fi]
            x_pt = torch.FloatTensor(x_perm).unsqueeze(0).to(device)
            with torch.no_grad():
                perm_out = float(model(x_pt).item())
            diffs.append(abs(base_out - perm_out))
        importances[fi] = np.mean(diffs)

    return importances


def batch_permutation_shap(
    model: nn.Module,
    X_samples: np.ndarray,
    X_background: np.ndarray,
    n_permutations: int = 50,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Compute permutation SHAP for multiple samples.

    Parameters
    ----------
    model : nn.Module
    X_samples : np.ndarray, shape (N, T, D)
    X_background : np.ndarray, shape (M, T, D)
    n_permutations : int
    device : torch.device or None

    Returns
    -------
    importances : np.ndarray, shape (N, D)
    """
    results = []
    for x in X_samples:
        imp = permutation_shap(model, x, X_background, n_permutations, device)
        results.append(imp)
    return np.stack(results)
