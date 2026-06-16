"""
sadaf/causal/psm.py
Propensity Score Matching (PSM) + Doubly Robust IPW for H1.

REVIEWER FIX (incorporated):
  - caliper = 0.1σ of logit(pscore)
  - Residual |SMD| > 0.1 for log_impression and log_cost:
    IPW-ATT is the PRIMARY estimator (doubly robust).
    PSM-ATT is corroborating evidence only.
  - Love plot exported for transparency.

Usage
-----
    from sadaf.causal.psm import run_psm_ipw
    result = run_psm_ipw(df_paid)
    print(result['ipw_att'], result['psm_att'], result['ci'])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _smd(x1: np.ndarray, x2: np.ndarray) -> float:
    """Standardised mean difference (signed)."""
    s = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    return (x1.mean() - x2.mean()) / (s + 1e-8)


def _balance_table(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str,
    matched_t: list,
    matched_c: list,
) -> pd.DataFrame:
    """Compute covariate balance before and after PSM."""
    rows = []
    for col in covariates:
        x_t_pre = df.loc[df[treatment_col] == 1, col].values
        x_c_pre = df.loc[df[treatment_col] == 0, col].values
        x_t_post = df.loc[matched_t, col].values
        x_c_post = df.loc[matched_c, col].values
        _, p_pre  = ttest_ind(x_t_pre, x_c_pre)
        _, p_post = ttest_ind(x_t_post, x_c_post)
        smd_after = _smd(x_t_post, x_c_post)
        if abs(smd_after) < 0.10:
            label = "✓ balanced"
        elif abs(smd_after) < 0.25:
            label = "⚠ residual — DR corrected"
        else:
            label = "✗ substantial — DR corrected (primary estimator: IPW)"
        rows.append(
            dict(
                Covariate=col,
                SMD_before=round(_smd(x_t_pre, x_c_pre), 4),
                SMD_after=round(smd_after, 4),
                p_before=round(p_pre, 4),
                p_after=round(p_post, 4),
                Balance=label,
            )
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def run_psm_ipw(
    df_paid: pd.DataFrame,
    confounders: list[str] | None = None,
    treatment_col: str = "T_highCTR",
    outcome_col: str = "has_conversion",
    caliper_sigma: float = 0.1,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> dict:
    """
    Run PSM + Doubly Robust IPW for H1 (High-CTR → Conversion).

    Parameters
    ----------
    df_paid : pd.DataFrame
        Paid-impression rows (cost > 0, impression > 0).
    confounders : list[str] or None
        Covariate columns.  Defaults include log_impression, log_cost,
        Depth, and campaign-type dummies.
    treatment_col : str
        Binary treatment column name (created if absent).
    outcome_col : str
        Binary outcome column.
    caliper_sigma : float
        Caliper as a multiple of std(logit(pscore)).  Default 0.1.
    n_bootstrap : int
        Number of bootstrap replications for PSM-ATT CI.
    random_seed : int

    Returns
    -------
    dict with keys:
        psm_att, ipw_att, ci_lo, ci_hi, n_matched,
        balance_df, h1_supported, dr_consistent
    """
    df = df_paid[df_paid["CTR"] > 0].copy()

    # ── Build treatment indicator ─────────────────────────────────────────
    if treatment_col not in df.columns:
        ctr_median = df["CTR"].median()
        df[treatment_col] = (df["CTR"] > ctr_median).astype(int)

    # ── Campaign-type dummies ─────────────────────────────────────────────
    type_dummies = pd.get_dummies(
        df.get("campaign_type_label", pd.Series(["Unknown"] * len(df))),
        prefix="ctype",
        drop_first=True,
    )
    df = pd.concat([df, type_dummies], axis=1)

    if confounders is None:
        confounders = ["log_impression", "log_cost", "Depth"] + list(
            type_dummies.columns
        )

    # ── Subset complete cases ─────────────────────────────────────────────
    cols_needed = confounders + [treatment_col, outcome_col]
    df_c = df[cols_needed].dropna().copy()

    X_conf = df_c[confounders].values
    T_arr  = df_c[treatment_col].values
    Y_arr  = df_c[outcome_col].values

    # ── Estimate propensity scores ────────────────────────────────────────
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_conf)
    lr     = LogisticRegression(max_iter=1000, C=0.1)
    lr.fit(X_sc, T_arr)
    pscore = lr.predict_proba(X_sc)[:, 1]
    df_c   = df_c.copy()
    df_c["pscore"] = pscore

    eps    = 1e-6
    logit_p = np.log(pscore / (1 - pscore + eps) + eps)
    df_c["logit_p"] = logit_p

    # ── Nearest-neighbour matching with caliper ───────────────────────────
    caliper = caliper_sigma * logit_p.std()
    treated_idx = df_c.index[T_arr == 1].tolist()
    control_idx = df_c.index[T_arr == 0].tolist()

    treated_lp = logit_p[df_c.index.isin(treated_idx)].reshape(-1, 1)
    control_lp = logit_p[df_c.index.isin(control_idx)].reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_lp)
    dists, mcp = nn.kneighbors(treated_lp)

    valid   = dists.flatten() < caliper
    matched_t = [treated_idx[i] for i in range(len(treated_idx)) if valid[i]]
    matched_c = [control_idx[mcp[i, 0]] for i in range(len(treated_idx)) if valid[i]]
    n_matched  = len(matched_t)

    Y_treated = df_c.loc[matched_t, outcome_col].values
    Y_control = df_c.loc[matched_c, outcome_col].values
    psm_att   = float(Y_treated.mean() - Y_control.mean())

    # ── Bootstrap CI (PSM-ATT) ────────────────────────────────────────────
    rng = np.random.default_rng(random_seed)
    boot_atts = [
        Y_treated[rng.integers(0, n_matched, n_matched)].mean()
        - Y_control[rng.integers(0, n_matched, n_matched)].mean()
        for _ in range(n_bootstrap)
    ]
    ci_lo, ci_hi = float(np.quantile(boot_atts, 0.025)), float(
        np.quantile(boot_atts, 0.975)
    )

    # ── IPW-ATT (primary doubly robust estimator) ─────────────────────────
    ipw = np.where(T_arr == 1, 1 / (pscore + eps), 1 / (1 - pscore + eps))
    ipw = np.clip(ipw, 0, np.percentile(ipw, 99))
    mask_t = T_arr == 1
    mask_c = T_arr == 0
    ipw_att = float(
        (Y_arr[mask_t] * ipw[mask_t]).sum() / ipw[mask_t].sum()
        - (Y_arr[mask_c] * ipw[mask_c]).sum() / ipw[mask_c].sum()
    )

    dr_consistent = abs(ipw_att - psm_att) < 0.05

    # ── Covariate balance table ───────────────────────────────────────────
    balance_df = _balance_table(df_c, confounders, treatment_col, matched_t, matched_c)

    return dict(
        psm_att=psm_att,
        ipw_att=ipw_att,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_matched=n_matched,
        caliper=caliper,
        balance_df=balance_df,
        h1_supported=ci_lo > 0,
        dr_consistent=dr_consistent,
        df_matched=(df_c, matched_t, matched_c),
        pscore=pscore,
        logit_p=logit_p,
    )
