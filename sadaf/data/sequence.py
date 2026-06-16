"""
sadaf/data/sequence.py
----------------------
Converts the raw hourly ad-performance DataFrame into fixed-length
sliding-window sequences suitable for recurrent model training.

Aggregation unit: (ad_group_id × Hours)
Sliding window  : seq_len look-back steps → 1 forecast step
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) sequence arrays from an hourly ad-performance DataFrame.

    Steps
    -----
    1. Aggregate to (ad_group_id × Hours) level.
    2. Engineer log/cyclical features.
    3. For each ad group with ≥ seq_len + 1 hourly observations, extract
       sliding windows of length seq_len; the label is the value at t + seq_len.

    Parameters
    ----------
    df_src : pd.DataFrame
        Input DataFrame (should already have cost > 0, etc. filtered as needed).
    target_col : str
        Column name of the prediction target (e.g. 'log_ROAS', 'has_roas').
    seq_len : int
        Look-back window length in hours.
    features : list of str
        Feature columns included in X.

    Returns
    -------
    X : np.ndarray, shape (N, seq_len, D)
    Y : np.ndarray, shape (N,)
    """
    # Exclude hour=0 to avoid midnight artefacts in the aggregation
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

    # Derived features needed by the feature list
    hagg["log_ROAS"]       = np.log1p(hagg["ROAS"])
    hagg["log_cost"]       = np.log1p(hagg["cost"])
    hagg["log_impression"] = np.log1p(hagg["impression"])
    hagg["hour_sin"]       = np.sin(2 * np.pi * hagg["Hours"] / 24)
    hagg["hour_cos"]       = np.cos(2 * np.pi * hagg["Hours"] / 24)
    hagg["has_roas"]       = (hagg["ROAS"] > 0).astype(float)

    seqs_X, seqs_Y = [], []
    for grp in hagg["ad_group_id"].unique():
        g = hagg[hagg["ad_group_id"] == grp].sort_values("Hours")
        if len(g) < seq_len + 1:
            continue
        X_arr = g[features].fillna(0).values
        Y_arr = g[target_col].fillna(0).values
        for i in range(len(g) - seq_len):
            seqs_X.append(X_arr[i : i + seq_len])
            seqs_Y.append(Y_arr[i + seq_len])

    if not seqs_X:
        D = len(features)
        return (
            np.zeros((0, seq_len, D), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    return (
        np.array(seqs_X, dtype=np.float32),
        np.array(seqs_Y, dtype=np.float32),
    )


# ── Train / val / test temporal split ─────────────────────────────────────────
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
    Chronological (non-shuffled) train / validation / test split.

    The sequences are assumed to be ordered by time. No random shuffling
    is applied, preserving the temporal dependency structure required by
    valid time-series evaluation.

    Parameters
    ----------
    X, Y : np.ndarray
        Sequence arrays of shape (N, seq_len, D) and (N,).
    train_frac : float
        Fraction of data used for training.
    val_frac : float
        Cumulative fraction at the end of the validation set
        (val_frac - train_frac is the validation size).

    Returns
    -------
    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te)
    """
    N    = len(X)
    n_tr = int(train_frac * N)
    n_va = int(val_frac * N)
    return (
        (X[:n_tr],    Y[:n_tr]),
        (X[n_tr:n_va], Y[n_tr:n_va]),
        (X[n_va:],    Y[n_va:]),
    )


# ── MinMax normalisation (fit on train, transform val/test) ────────────────────
def normalize_sequences(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit MinMaxScaler on training sequences, apply to val and test.
    Clipping is applied to val/test to handle out-of-range values.

    Returns
    -------
    X_tr_n, X_va_n, X_te_n : np.ndarray (normalised)
    scaler : fitted MinMaxScaler
    """
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
