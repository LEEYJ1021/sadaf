"""
sadaf/explainability/intgrad.py
Integrated Gradients attribution for SADAF (§7 / H5).

Method (Sundararajan et al. 2017):
  IG(x) = (x − x') × ∫₀¹ ∇f(x' + α(x − x')) dα

CPU fallback is used because gradient computation through the LSTM
across many samples is memory-intensive on GPU when combined with
the main training loop.

SADAF paper note (W4 reframing):
  IntGrad measures feature-level importance averaged over time steps —
  a different axis from Attention's temporal positional weighting.
  The three gradient-based methods (GS-SHAP, IntGrad, Perm-SHAP)
  achieve avg Spearman ρ ≥ 0.62 across all clusters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def integrated_gradients(
    model: nn.Module,
    x_sample: np.ndarray,
    baseline: np.ndarray | None = None,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients for a single input sequence.

    The model is temporarily set to training mode to enable gradient flow
    through MC-Dropout layers (consistent with SADAF's Bayesian setup).
    It is restored to eval mode after computation.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model accepting (1, T, D) input.
    x_sample : np.ndarray, shape (T, D)
        Input sequence to explain.
    baseline : np.ndarray or None, shape (T, D)
        Reference point. Defaults to all-zeros (zero baseline).
    n_steps : int
        Number of Riemann steps for integral approximation.

    Returns
    -------
    attribution : np.ndarray, shape (T, D)
        Signed feature × time attribution.
    """
    was_training = model.training

    # CPU fallback: move model to CPU for grad computation
    original_device = next(model.parameters()).device
    model_cpu = model.cpu()
    model_cpu.train()  # enable dropout for Bayesian models

    if baseline is None:
        baseline = np.zeros_like(x_sample)

    x_t = torch.FloatTensor(x_sample)
    b_t = torch.FloatTensor(baseline)

    grads = []
    alphas = np.linspace(0, 1, n_steps)
    for alpha in alphas:
        inp = (b_t + alpha * (x_t - b_t)).unsqueeze(0).requires_grad_(True)
        out = model_cpu(inp)
        model_cpu.zero_grad()
        out.sum().backward()
        if inp.grad is not None:
            grads.append(inp.grad.squeeze(0).detach().numpy().copy())
        else:
            grads.append(np.zeros_like(x_sample))

    # Restore model to original device and state
    model_cpu.to(original_device)
    if not was_training:
        model_cpu.eval()

    avg_grad     = np.mean(grads, axis=0)  # (T, D)
    attribution  = (x_sample - baseline) * avg_grad
    return attribution


def batch_integrated_gradients(
    model: nn.Module,
    X_samples: np.ndarray,
    baseline: np.ndarray | None = None,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients for multiple samples.

    Parameters
    ----------
    model : nn.Module
    X_samples : np.ndarray, shape (N, T, D)
    baseline : np.ndarray or None, shape (T, D)
    n_steps : int

    Returns
    -------
    attributions : np.ndarray, shape (N, T, D)
    """
    if baseline is None:
        baseline = X_samples.mean(axis=0)

    attributions = []
    for x in X_samples:
        attr = integrated_gradients(model, x, baseline=baseline, n_steps=n_steps)
        attributions.append(attr)
    return np.stack(attributions)
