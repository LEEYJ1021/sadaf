"""
sadaf/augmentation/copula.py
Gaussian Copula augmentation for sparse ad performance sequences.

Reference: SADAF §4 — FSD validation criterion (<2.0 accept).
"""

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import ks_2samp


def copula_augment(
    X_real: np.ndarray,
    n_synthetic: int,
    random_state: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """
    Generate synthetic sequences via Gaussian Copula.

    Procedure:
      1. Flatten (N, T, D) → (N, T*D).
      2. Marginal-normalise with QuantileTransformer.
      3. Fit multivariate Gaussian on normalised data.
      4. Sample, then invert QuantileTransformer.
      5. Reshape → (n_synthetic, T, D).

    Parameters
    ----------
    X_real : np.ndarray, shape (N, T, D)
        Real training sequences (MinMax-scaled or raw).
    n_synthetic : int
        Number of synthetic sequences to generate.
    random_state : int
        RNG seed for reproducibility.
    verbose : bool
        Print KS validation summary.

    Returns
    -------
    X_syn : np.ndarray, shape (n_synthetic, T, D)
    """
    N, T, D = X_real.shape
    X_flat = X_real.reshape(N, T * D)

    # Marginal normalisation to Gaussian via empirical CDF
    qt = QuantileTransformer(
        output_distribution="normal",
        random_state=random_state,
    )
    X_norm = qt.fit_transform(X_flat)

    # Empirical covariance (regularised for PSD)
    cov = np.cov(X_norm.T)
    cov = (cov + cov.T) / 2
    cov += 1e-5 * np.eye(cov.shape[0])

    # Sample from multivariate Gaussian copula
    rng = np.random.default_rng(random_state)
    X_syn_n = rng.multivariate_normal(
        mean=np.zeros(X_norm.shape[1]),
        cov=cov,
        size=n_synthetic,
    )

    # Invert marginal transform
    X_syn = qt.inverse_transform(X_syn_n)

    if verbose:
        n_check = min(T * D, 10)
        ks_vals = [
            ks_2samp(X_flat[:, d], X_syn[:, d])[0]
            for d in range(n_check)
        ]
        print(
            f"    Copula KS validation: "
            f"mean_KS={np.mean(ks_vals):.3f} "
            f"(lower = more realistic)"
        )

    return X_syn.reshape(n_synthetic, T, D)
