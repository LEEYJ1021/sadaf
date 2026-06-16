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
    rng = np.random.default_rng(seed)
    N = len(X_real)
    starts = np.arange(0, N - block_size + 1)

    syn_X, syn_Y = [], []
    for _ in range(n_synthetic):
        n_blocks = N // block_size + 1
        s = rng.choice(starts, size=n_blocks)
        idx = np.concatenate(
            [np.arange(si, min(si + block_size, N)) for si in s]
        )[:N]
        syn_X.append(X_real[idx])
        syn_Y.append(Y_real[idx])

    X_out = np.concatenate(syn_X[:n_synthetic])
    Y_out = np.concatenate(syn_Y[:n_synthetic])
    return X_out, Y_out
