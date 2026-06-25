"""
sadaf/training/trainer.py  [FIXED v2]
--------------------------------------
Changes vs. original
---------------------
FIX-1  (Learning-curve / train–val gap):
  - train_model() accepts an optional real_val_loader parameter.
    When supplied, early stopping is driven by real_val_loader while the
    history dict records *both* the augmented-distribution val loss
    (history["val_aug"]) and the real-data val loss (history["val_real"]).
    Figure 6 will then show three curves that make the domain gap explicit
    rather than misleading.
  - A domain_gap_report() helper computes the gap statistics for the
    paper text (§W3 / robustness).
  - All other functions (eval_reg, eval_cls, DM, bootstrap CI, smd)
    are unchanged.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import t as t_dist
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, mean_squared_error, r2_score,
)
from typing import Dict, Optional, Tuple

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
    val_loader: DataLoader,                   # augmented-dist val (original)
    epochs: int = TRAIN_EPOCHS,
    lr: float = TRAIN_LR,
    patience: int = TRAIN_PATIENCE,
    task: str = "reg",
    weight_decay: float = TRAIN_WD,
    verbose: bool = True,
    # ── [FIX-1] new parameter ──────────────────────────────────────────────
    real_val_loader: Optional[DataLoader] = None,
    # When provided: early stopping is based on REAL validation loss.
    # val_loader (augmented) is still recorded for diagnostic purposes.
    # When None: behaviour identical to original (early stop on val_loader).
) -> Tuple[nn.Module, Dict]:
    """
    Train with AdamW + CosineAnnealingLR + early stopping.

    [FIX-1] If real_val_loader is provided, early stopping is driven by
    the real held-out data loss, not the augmented-distribution val loss.
    This prevents the model from stopping too early due to the domain gap
    between synthetic training data and real validation data.

    Returns
    -------
    model   : best checkpoint (by real val loss if supplied, else aug val)
    history : dict with keys
                "train"    — train Huber/BCE per epoch
                "val"      — augmented-dist val loss per epoch
                "val_real" — real-data val loss per epoch  [FIX-1, may be []]
    """
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = (
        nn.BCEWithLogitsLoss() if task == "cls" else nn.HuberLoss()
    )

    best_val, best_state, patience_cnt = float("inf"), None, 0
    history = {"train": [], "val": [], "val_real": []}  # [FIX-1] added val_real

    for epoch in range(epochs):
        # ── training ──────────────────────────────────────────────────────
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

        # ── augmented-dist validation (always recorded) ────────────────────
        model.eval()
        va_loss = []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                va_loss.append(criterion(model(Xb), Yb).item())

        tm  = float(np.mean(tr_loss))
        vm  = float(np.mean(va_loss))
        history["train"].append(tm)
        history["val"].append(vm)

        # ── [FIX-1] real-data validation (used for early stopping) ─────────
        if real_val_loader is not None:
            rv_loss = []
            with torch.no_grad():
                for Xb, Yb in real_val_loader:
                    Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                    rv_loss.append(criterion(model(Xb), Yb).item())
            rvm = float(np.mean(rv_loss))
            history["val_real"].append(rvm)
            stopping_val = rvm          # early-stop on real val
        else:
            stopping_val = vm           # original behaviour

        scheduler.step()

        # ── early stopping ─────────────────────────────────────────────────
        if stopping_val < best_val:
            best_val     = stopping_val
            best_state   = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            if verbose:
                print(f"    Early stop @ epoch {epoch + 1}  "
                      f"(best {'real' if real_val_loader else 'aug'} "
                      f"val={best_val:.4f})")
            break

        if verbose and (epoch + 1) % 10 == 0:
            msg = f"    Ep {epoch + 1:3d}: train={tm:.4f}  val_aug={vm:.4f}"
            if real_val_loader is not None:
                msg += f"  val_real={rvm:.4f}"
            print(msg)

    model.load_state_dict(best_state)
    return model, history


# ── [FIX-1] Domain-gap diagnostics ────────────────────────────────────────────
def domain_gap_report(history: Dict, model_name: str = "") -> Dict[str, float]:
    """
    Compute train–val gap statistics from a history dict produced by
    train_model().

    Returns a dict with:
      best_epoch     — epoch index of best val checkpoint
      best_val_aug   — best augmented-dist val loss
      best_val_real  — best real val loss (nan if not recorded)
      final_train    — train loss at best epoch
      gap_aug        — best_val_aug - final_train  (classic overfit metric)
      gap_real       — best_val_real - final_train  (more meaningful gap)
    """
    tr   = np.array(history["train"])
    va   = np.array(history["val"])
    vr   = np.array(history["val_real"]) if history["val_real"] else None

    best_aug_ep  = int(np.argmin(va))
    best_real_ep = int(np.argmin(vr)) if vr is not None else best_aug_ep

    report = {
        "model":          model_name,
        "best_epoch_aug": best_aug_ep + 1,
        "best_val_aug":   float(va[best_aug_ep]),
        "final_train":    float(tr[best_aug_ep]),
        "gap_aug":        float(va[best_aug_ep] - tr[best_aug_ep]),
    }
    if vr is not None:
        report["best_epoch_real"] = best_real_ep + 1
        report["best_val_real"]   = float(vr[best_real_ep])
        report["gap_real"]        = float(vr[best_real_ep] - tr[best_real_ep])
    else:
        report["best_val_real"] = float("nan")
        report["gap_real"]      = float("nan")

    return report


# ── Regression evaluation ──────────────────────────────────────────────────────
def eval_reg(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
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
    d     = e1 ** 2 - e2 ** 2
    d_bar = d.mean()
    n     = len(d)
    gamma0 = np.var(d, ddof=1)
    gammas = [
        (np.cov(d[k:], d[:-k])[0, 1] if k > 0 else gamma0)
        for k in range(h)
    ]
    var_d = (gamma0 + 2 * sum(gammas[1:])) / n
    dm    = d_bar / np.sqrt(max(var_d, 1e-12))
    pv    = 2 * (1 - t_dist.cdf(abs(dm), df=n - 1))
    return float(dm), float(pv)


# ── Bootstrap RMSE CI ──────────────────────────────────────────────────────────
def bootstrap_rmse_ci(
    preds: np.ndarray,
    targets: np.ndarray,
    n_boot: int = DM_N_BOOT,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
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


# ── Standardized Mean Difference ───────────────────────────────────────────────
def smd(x1: np.ndarray, x2: np.ndarray) -> float:
    s = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    return float((x1.mean() - x2.mean()) / (s + 1e-8))
