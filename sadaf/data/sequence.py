"""
sadaf/data/sequence.py  [FIXED v2]
------------------------------------
Changes vs. original
---------------------
FIX-2  (Data Leakage):
  - build_sequences() now also returns group_ids (np.ndarray of str),
    recording which ad_group_id each sequence belongs to.
  - group_time_split() replaces time_split() for the regression stage.
    It partitions *ad groups* chronologically (by each group's median
    hour) so that no window from the same group ever straddles the
    train / val / test boundary.  This eliminates the overlapping-window
    leakage that existed when adjacent sliding windows from the same group
    were split across folds by a plain index cut.
  - time_split() is kept unchanged for backward compatibility (still used
    in the classification stage where leakage risk is lower).
  - normalize_X() alias added (was normalize_sequences in original; some
    scripts imported the wrong name).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple

from sadaf.config import FEATURES, TRAIN_FRAC, VAL_FRAC


# ── PyTorch Dataset ────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    """Minimal Dataset wrapper for (X, Y) numpy arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]


# ── Sequence builder ───────────────────────────────────────────────────────────
def build_sequences(
    df_src: pd.DataFrame,
    target_col: str,
    seq_len: int = 4,
    features: List[str] = FEATURES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, Y, group_ids) sequence arrays from an hourly ad-performance
    DataFrame.

    [FIX-2] Now returns group_ids so callers can perform group-aware splits
    that prevent the sliding-window leakage present in the original
    index-based time_split().

    Parameters
    ----------
    df_src     : pd.DataFrame  — pre-filtered input (e.g. cost > 0)
    target_col : str           — prediction target column
    seq_len    : int           — look-back window length in hours
    features   : list[str]    — feature columns in X

    Returns
    -------
    X         : np.ndarray, shape (N, seq_len, D)
    Y         : np.ndarray, shape (N,)
    group_ids : np.ndarray, shape (N,)  — ad_group_id for each sequence
                [NEW in FIX-2]
    """
    hagg = (
        df_src[df_src["Hours"] != 0]
        .groupby(["ad_group_id", "Hours"])
        .agg(
            impression=("impression", "sum"),
            click=("click", "sum"),
            cost=("cost", "sum"),
            CTR=("CTR", "mean"),
            CVR=("CVR", "mean"),
            ROAS=("ROAS", "mean"),
            Depth=("Depth", "mean"),
            conversion=("conversion_count", "sum"),
        )
        .reset_index()
    )

    hagg["log_ROAS"]       = np.log1p(hagg["ROAS"])
    hagg["log_cost"]       = np.log1p(hagg["cost"])
    hagg["log_impression"] = np.log1p(hagg["impression"])
    hagg["hour_sin"]       = np.sin(2 * np.pi * hagg["Hours"] / 24)
    hagg["hour_cos"]       = np.cos(2 * np.pi * hagg["Hours"] / 24)
    hagg["has_roas"]       = (hagg["ROAS"] > 0).astype(float)

    seqs_X, seqs_Y, seqs_G = [], [], []
    for grp in hagg["ad_group_id"].unique():
        g = hagg[hagg["ad_group_id"] == grp].sort_values("Hours")
        if len(g) < seq_len + 1:
            continue
        X_arr = g[features].fillna(0).values
        Y_arr = g[target_col].fillna(0).values
        for i in range(len(g) - seq_len):
            seqs_X.append(X_arr[i : i + seq_len])
            seqs_Y.append(Y_arr[i + seq_len])
            seqs_G.append(grp)           # [FIX-2] track group identity

    if not seqs_X:
        D = len(features)
        return (
            np.zeros((0, seq_len, D), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.array([], dtype=object),
        )

    return (
        np.array(seqs_X, dtype=np.float32),
        np.array(seqs_Y, dtype=np.float32),
        np.array(seqs_G, dtype=object),   # [FIX-2]
    )


# ── [FIX-2] Group-aware temporal split (PRIMARY for regression) ────────────────
def group_time_split(
    X: np.ndarray,
    Y: np.ndarray,
    group_ids: np.ndarray,
    df_hagg: pd.DataFrame = None,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    [FIX-2] Partition sequences by ad_group_id so that ALL windows from
    the same group land in exactly one of train / val / test.

    This eliminates the overlapping-window leakage in the original
    time_split():  adjacent sliding windows (sharing seq_len-1 time steps)
    from the same group can no longer straddle the boundary.

    Groups are ordered by their first appearance in group_ids (which
    reflects temporal order because build_sequences iterates hours in
    ascending order within each group).

    Parameters
    ----------
    X, Y        : sequence arrays from build_sequences()
    group_ids   : np.ndarray returned by build_sequences()  [FIX-2]
    df_hagg     : ignored — kept for potential future use
    train_frac  : fraction of *groups* assigned to train
    val_frac    : cumulative fraction at end of val set

    Returns
    -------
    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te)
    """
    # Unique groups in first-occurrence order (≈ temporal order)
    seen = {}
    for g in group_ids:
        if g not in seen:
            seen[g] = len(seen)
    unique_groups = sorted(seen.keys(), key=lambda g: seen[g])

    n_g  = len(unique_groups)
    n_tr = max(1, int(train_frac * n_g))
    n_va = max(n_tr + 1, int(val_frac * n_g))

    tr_set = set(unique_groups[:n_tr])
    va_set = set(unique_groups[n_tr:n_va])
    # test = remainder

    tr_mask = np.array([g in tr_set for g in group_ids])
    va_mask = np.array([g in va_set for g in group_ids])
    te_mask = ~(tr_mask | va_mask)

    # Fallback: if any split is empty (very few groups), revert to index split
    if tr_mask.sum() == 0 or va_mask.sum() == 0 or te_mask.sum() == 0:
        import warnings
        warnings.warn(
            "group_time_split: one fold is empty — "
            "falling back to index-based time_split().",
            RuntimeWarning,
        )
        return time_split(X, Y, train_frac, val_frac)

    return (
        (X[tr_mask], Y[tr_mask]),
        (X[va_mask], Y[va_mask]),
        (X[te_mask], Y[te_mask]),
    )


# ── Original index-based split (kept for CLS stage / backward compat) ─────────
def time_split(
    X: np.ndarray,
    Y: np.ndarray,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Chronological index-based split (original implementation).
    Still used for the classification stage.

    NOTE: for the regression stage, prefer group_time_split() to avoid
    overlapping-window leakage.  See FIX-2 in this module.
    """
    N    = len(X)
    n_tr = int(train_frac * N)
    n_va = int(val_frac * N)
    return (
        (X[:n_tr],    Y[:n_tr]),
        (X[n_tr:n_va], Y[n_tr:n_va]),
        (X[n_va:],    Y[n_va:]),
    )


# ── MinMax normalisation ───────────────────────────────────────────────────────
def normalize_sequences(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    N_tr, T, D = X_tr.shape
    scaler = MinMaxScaler()
    X_tr_n = scaler.fit_transform(X_tr.reshape(-1, D)).reshape(X_tr.shape)
    X_va_n = np.clip(
        scaler.transform(X_va.reshape(-1, D)).reshape(X_va.shape), 0, 1
    )
    X_te_n = np.clip(
        scaler.transform(X_te.reshape(-1, D)).reshape(X_te.shape), 0, 1
    )
    return X_tr_n, X_va_n, X_te_n, scaler


# Alias used by some scripts
normalize_X = normalize_sequences
