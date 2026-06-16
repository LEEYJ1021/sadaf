"""
sadaf/causal/moderation.py
OLS HC3-robust moderation analysis for H3.

H3: The positive CTR→ROAS relationship is moderated by campaign type,
    such that Search campaigns exhibit a steeper slope than Shopping.

Result (SADAF §7 / RQ3):
  β_interaction = 0.386  (p < 0.001)
  ME_Search  = 0.949
  ME_Shopping = 0.563
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def run_moderation(
    df_paid: pd.DataFrame,
    outcome: str = "log_ROAS",
    treatment: str = "log_CTR",
    moderator: str = "is_search",
    covariates: list[str] | None = None,
) -> dict:
    """
    HC3-robust OLS interaction model: CTR × campaign_type → ROAS.

    Parameters
    ----------
    df_paid : pd.DataFrame
        Paid rows with ROAS > 0, CTR > 0, campaign_type_label in data.
    outcome : str
        Dependent variable column.
    treatment : str
        Main treatment column.
    moderator : str
        Binary moderator (1 = Search, 0 = Shopping).
    covariates : list[str]
        Additional controls. Default: ['log_cost', 'log_impression'].

    Returns
    -------
    dict with keys:
        beta_ctr, beta_interaction, me_shopping, me_search,
        p_interaction, h3_supported, model_summary, model
    """
    if covariates is None:
        covariates = ["log_cost", "log_impression"]

    df = df_paid[
        (df_paid["CTR"] > 0)
        & (df_paid["ROAS"] > 0)
        & (df_paid["campaign_type_label"].isin(["Search", "Shopping"]))
    ].copy()

    if outcome not in df.columns:
        df["log_ROAS"] = np.log1p(df["ROAS"])
    if treatment not in df.columns:
        df["log_CTR"] = np.log1p(df["CTR"])
    if moderator not in df.columns:
        df["is_search"] = (df["campaign_type_label"] == "Search").astype(int)

    controls = " + ".join(covariates) if covariates else ""
    formula = (
        f"{outcome} ~ {treatment} * {moderator}"
        + (f" + {controls}" if controls else "")
    )

    model = smf.ols(formula, data=df).fit(cov_type="HC3")

    beta_ctr    = float(model.params[treatment])
    beta_int    = float(model.params[f"{treatment}:{moderator}"])
    me_shopping = beta_ctr
    me_search   = beta_ctr + beta_int
    p_int       = float(model.pvalues[f"{treatment}:{moderator}"])

    return dict(
        beta_ctr=beta_ctr,
        beta_interaction=beta_int,
        me_shopping=me_shopping,
        me_search=me_search,
        p_interaction=p_int,
        h3_supported=p_int < 0.05,
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
        model=model,
        df_used=df,
    )
