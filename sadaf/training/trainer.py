"""
sadaf/training/trainer.py
-------------------------
Generic training loop, evaluation functions, and statistical tests
shared across all SADAF model experiments.

Functions
---------
train_model           — AdamW + CosineAnnealing + early stopping
eval_reg              — regression evaluation (RMSE / MAE / R²)
eval_cls              — classification evaluation (AUC / F1 / AP)
find_best_threshold   — F1-optimal decision threshold on validation set
diebold_mariano       — DM test with HAC variance correction
bootstrap_rmse_ci     — bootstrap 95% CI for RMSE
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import t as t_dist
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, mean_squared_error, r2_score,
)
from typing import Dict, Tuple

from sadaf.config import (
    DEVICE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_WD,
    TRAIN_PATIENCE,
    DM_H,
    DM_N_BOOT,
    RANDOM_SEED,
)


# ── Training loop ──────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = TRAIN_EPOCHS,
    lr: float = TRAIN_LR,
    patience: int = TRAIN_PATIENCE,
    task: str = "reg",
    weight_decay: float = TRAIN_WD,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Train a model with AdamW, cosine LR decay, and early stopping.

    Parameters
    ----------
    model : nn.Module
        Uninitialised or pre-initialised model; moved to DEVICE.
    train_loader : DataLoader
    val_loader : DataLoader
    epochs : int
    lr : float
    patience : int
        Early stopping patience (epochs with no improvement).
    task : str
        "reg"  → Huber loss
        "cls"  → BCEWithLogitsLoss
    weight_decay : float
    verbose : bool
        Print loss every 10 epochs if True.

    Returns
    -------
    model : nn.Module
        Model loaded with best validation checkpoint.
    history : dict
        {"train": [...], "val": [...]} Huber / BCE loss per epoch.
    """
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion  = (
        nn.BCEWithLogitsLoss() if task == "cls" else nn.HuberLoss()
    )

    best_val, best_state, patience_cnt = float("inf"), None, 0
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        # ── training step ──────────────────────────────────────────────────
        model.train()
        tr_loss = []
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), Yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss.append(loss.item())

        # ── validation step ────────────────────────────────────────────────
        model.eval()
        va_loss = []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                va_loss.append(criterion(model(Xb), Yb).item())

        tm, vm = np.mean(tr_loss), np.mean(va_loss)
        history["train"].append(tm)
        history["val"].append(vm)
        scheduler.step()

        # ── early stopping ─────────────────────────────────────────────────
        if vm < best_val:
            best_val   = vm
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            if verbose:
                print(f"    Early stop @ epoch {epoch + 1}")
            break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Ep {epoch + 1:3d}: train={tm:.4f}  val={vm:.4f}")

    model.load_state_dict(best_state)
    return model, history


# ── Regression evaluation ──────────────────────────────────────────────────────
def eval_reg(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a regression model.

    Returns
    -------
    metrics : dict with keys RMSE, MAE, R2
    preds   : np.ndarray, shape (N,)
    targets : np.ndarray, shape (N,)
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for Xb, Yb in loader:
            preds.append(model(Xb.to(DEVICE)).cpu().numpy())
            targets.append(Yb.numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    metrics = {
        "RMSE": float(np.sqrt(np.mean((p - t) ** 2))),
        "MAE":  float(np.mean(np.abs(p - t))),
        "R2":   float(1 - np.sum((p - t) ** 2)
                      / (np.sum((t - t.mean()) ** 2) + 1e-8)),
    }
    return metrics, p, t


# ── Classification evaluation ──────────────────────────────────────────────────
def eval_cls(
    model: nn.Module,
    loader: DataLoader,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a binary classifier.

    Returns
    -------
    metrics : dict with keys AUC, F1, Acc, AP
    probs   : np.ndarray, shape (N,)   — sigmoid probabilities
    targets : np.ndarray, shape (N,)
    """
    model.eval()
    logits_all, targets_all = [], []
    with torch.no_grad():
        for Xb, Yb in loader:
            logits_all.append(model(Xb.to(DEVICE)).cpu().numpy())
            targets_all.append(Yb.numpy())
    logits  = np.concatenate(logits_all)
    targets = np.concatenate(targets_all)
    probs   = 1 / (1 + np.exp(-logits))
    preds   = (probs > threshold).astype(int)
    metrics = {
        "AUC": float(roc_auc_score(targets, probs)),
        "F1":  float(f1_score(targets, preds, zero_division=0)),
        "Acc": float(accuracy_score(targets, preds)),
        "AP":  float(average_precision_score(targets, probs)),
    }
    return metrics, probs, targets


def find_best_threshold(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Sweep thresholds in [0.1, 0.9] and return the value that maximises F1
    on the provided DataLoader (should be the validation set).

    Returns
    -------
    best_threshold : float
    best_f1        : float
    probs          : np.ndarray
    targets        : np.ndarray
    """
    model.eval()
    logits_all, targets_all = [], []
    with torch.no_grad():
        for Xb, Yb in loader:
            logits_all.append(model(Xb.to(DEVICE)).cpu().numpy())
            targets_all.append(Yb.numpy())
    logits  = np.concatenate(logits_all)
    targets = np.concatenate(targets_all)
    probs   = 1 / (1 + np.exp(-logits))

    best_t, best_f1 = 0.5, 0.0
    for thr in np.arange(0.1, 0.9, 0.02):
        f = f1_score(targets, (probs > thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, thr
    return best_t, best_f1, probs, targets


# ── Diebold-Mariano test ───────────────────────────────────────────────────────
def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = DM_H,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy with HAC variance.

    H0: E[d_t] = 0, where d_t = L(e1_t) − L(e2_t), L = squared loss.
    A negative DM statistic means model 1 is more accurate than model 2.

    Parameters
    ----------
    e1, e2 : np.ndarray
        Forecast errors for the two competing models.
    h : int
        Forecast horizon (default 1).

    Returns
    -------
    dm_stat : float
    p_value : float (two-sided)
    """
    d     = e1 ** 2 - e2 ** 2
    d_bar = d.mean()
    n     = len(d)
    gamma0 = np.var(d, ddof=1)
    gammas = [
        (np.cov(d[k:], d[:-k])[0, 1] if k > 0 else gamma0)
        for k in range(h)
    ]
    var_d  = (gamma0 + 2 * sum(gammas[1:])) / n
    dm     = d_bar / np.sqrt(max(var_d, 1e-12))
    pv     = 2 * (1 - t_dist.cdf(abs(dm), df=n - 1))
    return float(dm), float(pv)


# ── Bootstrap RMSE confidence interval ────────────────────────────────────────
def bootstrap_rmse_ci(
    preds: np.ndarray,
    targets: np.ndarray,
    n_boot: int = DM_N_BOOT,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap 95% CI for RMSE.

    Parameters
    ----------
    preds, targets : np.ndarray
    n_boot : int
    alpha : float
        Two-sided coverage level (default 0.05 → 95% CI).
    seed : int

    Returns
    -------
    rmse_point : float
    ci_lo      : float
    ci_hi      : float
    """
    rng  = np.random.default_rng(seed)
    n    = len(targets)
    rmse_point = float(np.sqrt(np.mean((preds - targets) ** 2)))
    boot_rmses = [
        float(np.sqrt(np.mean(
            (preds[idx := rng.integers(0, n, n)]
             - targets[idx]) ** 2
        )))
        for _ in range(n_boot)
    ]
    ci_lo = float(np.quantile(boot_rmses, alpha / 2))
    ci_hi = float(np.quantile(boot_rmses, 1 - alpha / 2))
    return rmse_point, ci_lo, ci_hi


# ── Standardized Mean Difference (covariate balance) ──────────────────────────
def smd(x1: np.ndarray, x2: np.ndarray) -> float:
    """Standardized Mean Difference for Love plot."""
    s = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    return float((x1.mean() - x2.mean()) / (s + 1e-8))
