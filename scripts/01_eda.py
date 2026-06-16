"""
01_eda.py — Exploratory Data Analysis for SADAF

Loads the raw ad-performance Excel file, applies basic preprocessing
(missing-value handling, derived features, log transforms, campaign-type
labeling), and prints summary diagnostics used to motivate the SADAF
research design (cold-start sparsity, zero-inflation, campaign mix).

Usage:
    python scripts/01_eda.py --data_path data/3월성과데이터(샘플).xlsx
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from sadaf.data.loader import load_and_preprocess

warnings.filterwarnings("ignore")


def run_eda(data_path: str) -> None:
    print(f"Loading data from: {data_path}")
    df, df_paid, df_roas = load_and_preprocess(data_path)

    print("\n=== 데이터 크기 ===")
    print(f"행: {df.shape[0]:,} / 열: {df.shape[1]}")

    print("\n=== 컬럼 목록 ===")
    for i, col in enumerate(df.columns):
        print(f"  {i + 1}. {col} ({df[col].dtype})")

    print("\n=== 결측값 ===")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing if len(missing) > 0 else "결측값 없음")

    print("\n=== 기술통계 ===")
    desc_cols = ["impression", "click", "cost", "CTR", "CVR", "ROAS",
                 "Depth", "conversion_count", "sales_by_conversion"]
    desc = df[desc_cols].describe().T
    desc["cv"] = desc["std"] / desc["mean"]
    print(desc.round(3))

    print("\n── Dataset summary ──────────────────────────────")
    print(f"  Total rows    : {len(df):>8,}")
    print(f"  Paid rows     : {len(df_paid):>8,}")
    print(f"  ROAS > 0      : {len(df_roas):>8,}  "
          f"({len(df_roas) / len(df_paid) * 100:.1f}% of paid)")
    print(f"  Conversion %  : {df['has_conversion'].mean() * 100:.2f}%")
    print(f"  ROAS=0 (spare): {(df_paid['ROAS'] == 0).mean() * 100:.1f}%  "
          f"← cold-start context, not a flaw")

    print("\n=== 캠페인 유형 분포 ===")
    print(df["campaign_type_label"].value_counts())

    print("\n=== 캠페인별 성과 요약 (상위 10, by cost) ===")
    campaign = df_paid.groupby("campaign_id").agg(
        impression=("impression", "sum"),
        click=("click", "sum"),
        cost=("cost", "sum"),
        CTR=("CTR", "mean"),
        CVR=("CVR", "mean"),
        ROAS=("ROAS", "mean"),
        conversion=("conversion_count", "sum"),
        ad_count=("ad_id", "nunique"),
    ).reset_index()
    print(campaign.sort_values("cost", ascending=False).head(10).to_string(index=False))

    print("\n✅ EDA complete.")


def main():
    parser = argparse.ArgumentParser(description="SADAF: Exploratory Data Analysis")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx",
                         help="Path to the raw ad-performance Excel file")
    args = parser.parse_args()
    run_eda(args.data_path)


if __name__ == "__main__":
    main()