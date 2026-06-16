"""
sadaf/causal/mediation.py
Baron-Kenny mediation analysis with bootstrap CI for H2.

Key finding (SADAF §6 / RQ2):
  a < 0 (high-CTR ads reduce Depth),
  b < 0 (deeper Depth reduces conversion),
  a×b > 0 → NEGATIVE SUPPRESSOR structure.

The proportion mediated (-42.8%) is reported in appendix only;
the sign pattern of paths a and b is the primary theoretical finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


def run_mediation(
    df_paid: pd.DataFrame,
    treatment_col: str = "log_CTR",
    mediator_col: str = "Depth",
    outcome_col: str = "has_conversion",
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> dict:
    """
    Baron-Kenny mediation decomposition (H2: CTR → Depth → Conversion).

    Parameters
    ----------
    df_paid : pd.DataFrame
        Paid-impression rows with CTR > 0 and Depth > 0.
    treatment_col : str
        Log-transformed treatment (default: log_CTR).
    mediator_col : str
        Mediator variable (default: Depth).
    outcome_col : str
        Binary outcome (default: has_conversion).
    n_bootstrap : int
        Bootstrap replications for CI on indirect path.
    random_seed : int

    Returns
    -------
    dict with keys:
        a, b, c_prime, c_total, indirect, ci_lo, ci_hi,
        suppressor_type, suppressor_interp, prop_mediated
    """
    df = df_paid[
        (df_paid["CTR"] > 0) & (df_paid["Depth"] > 0)
    ].copy()

    if treatment_col not in df.columns:
        df["log_CTR"] = np.log1p(df["CTR"])

    X = df[treatment_col].values.reshape(-1, 1)
    M = df[mediator_col].values
    Y = df[outcome_col].values

    # ── Path a: X → M ────────────────────────────────────────────────────
    lr_a = LinearRegression().fit(X, M)
    a    = float(lr_a.coef_[0])

    # ── Path b (and c'): X + M → Y ───────────────────────────────────────
    XM   = np.hstack([X, M.reshape(-1, 1)])
    lr_b = LogisticRegression(max_iter=500).fit(XM, Y)
    b_coef  = float(lr_b.coef_[0][1])
    c_prime = float(lr_b.coef_[0][0])

    # ── Total effect c: X → Y ────────────────────────────────────────────
    lr_c   = LogisticRegression(max_iter=500).fit(X, Y)
    c_total = float(lr_c.coef_[0][0])

    indirect = a * b_coef

    # ── Bootstrap CI on indirect path ────────────────────────────────────
    rng = np.random.default_rng(random_seed)
    boot_ind = []
    n = len(df)
    for _ in range(n_bootstrap):
        idx_b = rng.integers(0, n, n)
        Xb, Mb, Yb = X[idx_b], M[idx_b], Y[idx_b]
        a_b = LinearRegression().fit(Xb, Mb).coef_[0]
        try:
            b_b = LogisticRegression(max_iter=300).fit(
                np.hstack([Xb, Mb.reshape(-1, 1)]), Yb
            ).coef_[0][1]
            boot_ind.append(a_b * b_b)
        except Exception:
            pass

    ci_lo_m, ci_hi_m = float(np.quantile(boot_ind, 0.025)), float(
        np.quantile(boot_ind, 0.975)
    )

    # ── Classify suppressor type ──────────────────────────────────────────
    if a < 0 and b_coef < 0 and indirect > 0:
        suppressor_type = "Negative suppressor (a<0, b<0, a×b>0)"
        suppressor_interp = (
            "High-CTR ads reduce browsing depth (a<0): immediate-click "
            "campaigns bypass deliberate browsing. Among ads generating "
            "depth, deeper browsing reduces conversion (b<0), indicating "
            "depth proxies decision hesitancy, not engagement. "
            "The positive indirect product constitutes a negative suppressor."
        )
    elif a > 0 and b_coef > 0:
        suppressor_type  = "Consistent (positive) mediation"
        suppressor_interp = "Both paths positive; standard partial mediation."
    else:
        suppressor_type   = "Inconsistent mediation"
        suppressor_interp = "Mixed sign paths; interpret via path diagram."

    prop_mediated = indirect / (c_total + 1e-8) * 100

    return dict(
        a=a,
        b=b_coef,
        c_prime=c_prime,
        c_total=c_total,
        indirect=indirect,
        ci_lo=ci_lo_m,
        ci_hi=ci_hi_m,
        suppressor_type=suppressor_type,
        suppressor_interp=suppressor_interp,
        prop_mediated=prop_mediated,
        h2_supported=(ci_lo_m > 0),
        boot_indirect=np.array(boot_ind),
    )
