"""
sadaf/augmentation/mbb.py
Moving Block Bootstrap (MBB) for temporal ad performance sequences.

MBB preserves local autocorrelation structure by resampling contiguous
blocks rather than individual observations.  This is critical for ad
performance data, where hour-to-hour patterns (rush hours, night-time
drops) carry temporal dependency.

Reference: Kunsch (1989); SADAF §4.
"""

import numpy as np


def mbb_augment(
    X_real: np.ndarray,
    Y_real: np.ndarray,
    block_size: int = 3,
    n_synthetic: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (X, Y) pairs via Moving Block Bootstrap.

    Parameters
    ----------
    X_real : np.ndarray, shape (N, T, D)
        Real training sequences.
    Y_real : np.ndarray, shape (N,)
        Corresponding regression targets.
    block_size : int
        Number of consecutive samples per resampled block.
    n_synthetic : int
        Target number of synthetic samples (approximate).
    seed : int
        RNG seed.

    Returns
    -------
    X_mbb : np.ndarray, shape (~n_synthetic, T, D)
    Y_mbb : np.ndarray, shape (~n_synthetic,)
    """
    # [BUG-FIX] Prior implementation appended X_real[idx] — shape (N, T, D) —
    # on every loop iteration, then np.concatenate produced n_synthetic × N
    # rows instead of n_synthetic rows.  With N=174, n_synthetic=232 this
    # gave ~40,368 sequences instead of 232, causing the "174 → ~41180"
    # augmentation blow-up seen in the pipeline log.
    #
    # Fix: build the block-resampled index pool (length N) as before, then
    # select a single sequence from that pool per iteration, so each loop
    # iteration contributes exactly one (T, D) sequence.
    rng = np.random.default_rng(seed)
    N, T, D = X_real.shape
    starts = np.arange(0, N - block_size + 1)

    syn_X, syn_Y = [], []
    for _ in range(n_synthetic):
        n_blocks = N // block_size + 1
        s = rng.choice(starts, size=n_blocks)
        idx = np.concatenate(
            [np.arange(si, min(si + block_size, N)) for si in s]
        )[:N]
        # Select ONE sequence from the block-resampled pool → shape (T, D)
        pick = rng.integers(0, N)
        syn_X.append(X_real[idx[pick]])   # (T, D)
        syn_Y.append(Y_real[idx[pick]])

    X_out = np.stack(syn_X)   # (n_synthetic, T, D)
    Y_out = np.array(syn_Y)   # (n_synthetic,)
    return X_out, Y_out
