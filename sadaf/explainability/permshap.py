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
    seed: int | None = None,
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
    seed : int or None
        RNG seed.  Pass None (default) for truly independent draws across
        calls — critical when explaining multiple x_samples so that each
        call uses a different background-index sequence.  Passing a fixed
        integer makes a single call reproducible but will cause all calls
        to produce identical importance vectors if used repeatedly, which
        collapses cluster-level diversity and triggers "near-constant"
        warnings in agreement.py.

    Returns
    -------
    importances : np.ndarray, shape (D,)
        Mean |Δoutput| per feature dimension.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    T, D = x_sample.shape
    # [BUG-FIX] Previously hard-coded `rng = np.random.default_rng(42)`
    # caused every call to draw the same background indices, making the
    # importance vector identical across all x_samples (std ≈ 0 per cluster)
    # and triggering "near-constant → Spearman undefined" in agreement.py.
    rng = np.random.default_rng(seed)

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
    base_seed: int | None = None,
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
    base_seed : int or None
        If provided, sample i uses seed base_seed + i so each call is
        reproducible yet independent across samples.

    Returns
    -------
    importances : np.ndarray, shape (N, D)
    """
    results = []
    for i, x in enumerate(X_samples):
        seed = (base_seed + i) if base_seed is not None else None
        imp = permutation_shap(model, x, X_background, n_permutations, device, seed=seed)
        results.append(imp)
    return np.stack(results)
