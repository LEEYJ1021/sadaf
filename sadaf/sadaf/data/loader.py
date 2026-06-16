"""
sadaf/data/loader.py
--------------------
Data loading, cleaning, feature engineering, and subsetting helpers.

Expected input: Excel file with columns matching the schema described in
README.md (Date, Hours, customer_id, campaign_id, ad_group_id, ad_id,
impression, click, cost, sum_of_ad_rank, conversion_count,
sales_by_conversion, CTR, CVR, ROAS, Depth).
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ── Campaign type mapping ──────────────────────────────────────────────────────
_TYPE_MAP = {"01": "Search", "02": "Shopping", "04": "Zero-cost"}


def load_and_preprocess(
    path: str,
    sheet_name: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the ad performance Excel file and return three DataFrames:

    Parameters
    ----------
    path : str
        Path to the Excel file (e.g. "data/ad_performance.xlsx").
    sheet_name : int or str
        Sheet to read. Default 0 (first sheet).

    Returns
    -------
    df : pd.DataFrame
        Full dataset with all engineered features.
    df_paid : pd.DataFrame
        Subset where cost > 0 AND impression > 0.
    df_roas : pd.DataFrame
        Subset of df_paid where ROAS > 0.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    # ── Missing value imputation ──────────────────────────────────────────────
    # CTR and Depth are NaN only when impression == 0; fill with 0.
    df["CTR"]   = df["CTR"].fillna(0)
    df["Depth"] = df["Depth"].fillna(0)

    # ── Derived metrics ────────────────────────────────────────────────────────
    df["CPC"] = df.apply(
        lambda r: r["cost"] / r["click"] if r["click"] > 0 else 0, axis=1
    )
    df["CPA"] = df.apply(
        lambda r: r["cost"] / r["conversion_count"]
        if r["conversion_count"] > 0 else 0,
        axis=1,
    )
    df["has_conversion"] = (df["conversion_count"] > 0).astype(int)

    # ── Log-transformations (variance stabilisation) ───────────────────────────
    for col in [
        "impression", "click", "cost", "CTR", "CVR",
        "ROAS", "CPC", "conversion_count",
    ]:
        df[f"log_{col}"] = np.log1p(df[col])

    # ── Cyclical hour encoding ─────────────────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * df["Hours"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hours"] / 24)

    # ── Campaign type label ───────────────────────────────────────────────────
    df["campaign_type"] = df["campaign_id"].str.extract(r"a001-(\d+)-")
    df["campaign_type_label"] = (
        df["campaign_type"].map(_TYPE_MAP).fillna("Other")
    )

    # ── Hour-of-day bins (human-readable) ─────────────────────────────────────
    df["hour_bin"] = pd.cut(
        df["Hours"],
        bins=[0, 6, 12, 18, 24],
        labels=["Night(0-6)", "Morning(6-12)", "Afternoon(12-18)", "Evening(18-24)"],
        right=False,
        include_lowest=True,
    )

    # ── Subsets ────────────────────────────────────────────────────────────────
    df_paid = df[(df["cost"] > 0) & (df["impression"] > 0)].copy()
    df_roas = df_paid[df_paid["ROAS"] > 0].copy()

    _print_summary(df, df_paid, df_roas)
    return df, df_paid, df_roas


def _print_summary(df, df_paid, df_roas) -> None:
    print("── Dataset summary ──────────────────────────────────────")
    print(f"  Total rows    : {len(df):>8,}")
    print(f"  Paid rows     : {len(df_paid):>8,}")
    print(f"  ROAS > 0      : {len(df_roas):>8,}  "
          f"({len(df_roas) / max(len(df_paid), 1) * 100:.1f}% of paid)")
    print(f"  Conversion %  : {df['has_conversion'].mean() * 100:.2f}%")
    print(f"  Zero-ROAS %   : {(df_paid['ROAS'] == 0).mean() * 100:.1f}%")
    print("─────────────────────────────────────────────────────────")


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a descriptive statistics table (with coefficient of variation)
    for the main performance metrics.
    """
    cols = [
        "impression", "click", "cost", "CTR", "CVR",
        "ROAS", "Depth", "conversion_count", "sales_by_conversion",
    ]
    desc = df[cols].describe().T
    desc["cv"] = desc["std"] / desc["mean"].replace(0, np.nan)
    return desc.round(3)
