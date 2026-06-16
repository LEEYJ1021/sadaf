"""
04_augmentation.py — Sequence construction + β-VAE/Copula/MBB augmentation
pipeline with Fréchet Score Distance (FSD) validation.

Usage:
    python scripts/04_augmentation.py --data_path data/3월성과데이터(샘플).xlsx
"""
import argparse
import warnings

import numpy as np

from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import build_sequences, time_split, normalize_X
from sadaf.augmentation.pipeline import augment_pipeline
from sadaf.models.gru import GRUForecaster
from sadaf.training.trainer import train_model, SeqDataset
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURES = ["CTR", "CVR", "Depth", "log_cost",
            "log_impression", "hour_sin", "hour_cos"]


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: augmentation pipeline + FSD validation")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--target_n", type=int, default=800,
                         help="Target augmented training set size")
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)

    print("\n══ Sequence Dataset Construction ═══════════════════")
    X_reg, Y_reg = build_sequences(df_roas, "log_ROAS", FEATURES, seq_len=4)
    print(f"  REG sequences (SL=4): {X_reg.shape}")

    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te) = time_split(X_reg, Y_reg)
    MIN_VAL = 10
    if len(X_va) < MIN_VAL or len(X_te) < MIN_VAL:
        (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te) = time_split(
            X_reg, Y_reg, train_frac=0.60, val_frac=0.80)
    print(f"  Split — train/val/test: {len(X_tr)}/{len(X_va)}/{len(X_te)}")

    X_tr_n, X_va_n, X_te_n, _ = normalize_X(X_tr, X_va, X_te)

    print("\n══ Reference GRU (for FSD embedding) ═══════════════")
    D_in = len(FEATURES)
    bs = min(32, len(X_tr_n))
    tr_loader = DataLoader(SeqDataset(X_tr_n, Y_tr), batch_size=bs, shuffle=True)
    va_loader = DataLoader(SeqDataset(X_va_n, Y_va), batch_size=bs)
    ref_gru = GRUForecaster(D_in)
    ref_gru, _ = train_model(ref_gru, tr_loader, va_loader, epochs=50, patience=8)

    print("\n══ Augmentation Pipeline ════════════════════════════")
    X_aug, Y_aug = augment_pipeline(
        X_tr.astype(np.float32), Y_tr.astype(np.float32),
        target_n=max(args.target_n, len(X_tr) * 5), ref_lstm=ref_gru)

    print(f"\n  Final augmented set: {X_aug.shape[0]} sequences "
          f"(from {len(X_tr)} real)")
    print("\n✅ Augmentation pipeline complete.")


if __name__ == "__main__":
    main()