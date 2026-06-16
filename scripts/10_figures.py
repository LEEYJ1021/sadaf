"""
10_figures.py — Generate the full SADAF figure catalogue (Figures 1-14
and W1a-W7) by orchestrating the upstream analysis modules and saving
each output to figures/.

This script does not duplicate the statistical logic from scripts 01-09;
it imports their reusable functions and calls dedicated plotting
routines in sadaf.viz, writing all outputs to the figures/ directory.

Usage:
    python scripts/10_figures.py --data_path data/3월성과데이터(샘플).xlsx --out_dir figures/
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")

from sadaf.data.loader import load_and_preprocess
from sadaf.viz import main_figures, weakness_figures

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: generate full figure catalogue")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--out_dir", type=str, default="figures/")
    parser.add_argument("--skip_weakness", action="store_true",
                         help="Skip the W-series weakness-supplement figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, df_paid, df_roas = load_and_preprocess(args.data_path)

    print("══ Generating main figures (Fig. 1–14) ══════════════")
    main_figures.generate_all(df, df_paid, df_roas, out_dir=args.out_dir)

    if not args.skip_weakness:
        print("\n══ Generating weakness-supplement figures (W1–W7) ══")
        weakness_figures.generate_all(df, df_paid, df_roas, out_dir=args.out_dir)

    print(f"\n✅ All figures written to {args.out_dir}")


if __name__ == "__main__":
    main()