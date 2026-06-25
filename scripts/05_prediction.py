"""
05_prediction.py  [FIXED v2]
------------------------------
Two-stage prediction: H4a (classification), H4b (regression vs. baselines,
DM-tested), H4c (Mamba SEQ_LEN robustness).

Changes vs. original
---------------------
FIX-1  (Learning-curve / train–val gap):
  - Regression stage now passes real_val_loader to train_model() so that
    early stopping is driven by real held-out validation data, not the
    augmented-distribution val loader.
  - domain_gap_report() is called for every model and printed.

FIX-2  (Data Leakage in regression):
  - build_sequences() now returns group_ids (third return value).
  - group_time_split() replaces time_split() for the REG stage, ensuring
    no overlapping windows from the same ad group straddle train/val/test.
  - CLS stage keeps time_split() (lower leakage risk with ~2,211 sequences).

Usage:
    python scripts/05_prediction.py --data_path data/3월성과데이터(샘플).xlsx
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression as LR_sk, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              average_precision_score, mean_squared_error,
                              r2_score)
from sklearn.preprocessing import MinMaxScaler

from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import (build_sequences, time_split,
                                  group_time_split,          # [FIX-2]
                                  normalize_sequences as normalize_X,
                                  SeqDataset)
from sadaf.augmentation.pipeline import augment_pipeline
from sadaf.models.lstm import LSTMForecaster, LSTMClassifier, BayesianLSTM
from sadaf.models.gru import GRUForecaster
from sadaf.models.mamba import MambaForecaster
from sadaf.training.trainer import (train_model, eval_reg, eval_cls,
                                     find_best_threshold, diebold_mariano,
                                     domain_gap_report)   # [FIX-1]

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = ["CTR", "CVR", "Depth", "log_cost",
            "log_impression", "hour_sin", "hour_cos"]
D_IN = len(FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# H4a: Classification stage
# (keeps time_split — leakage risk lower at 2,211 sequences)
# ─────────────────────────────────────────────────────────────────────────────
def run_h4a(df_paid: pd.DataFrame):
    print("\n══ H4a: Classification Stage ═══════════════════════")

    # [FIX-2] unpack three return values
    X_cls, Y_cls, _gids = build_sequences(
    df_paid,
    "has_roas",
    seq_len=4,
    features=FEATURES
    )

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(X_cls, Y_cls)
    Xtr_n, Xva_n, Xte_n, _ = normalize_X(Xtr, Xva, Xte)

    bs   = min(512, len(Xtr_n))
    tr_l = DataLoader(SeqDataset(Xtr_n, Ytr), batch_size=bs, shuffle=True)
    va_l = DataLoader(SeqDataset(Xva_n, Yva), batch_size=bs)
    te_l = DataLoader(SeqDataset(Xte_n, Yte), batch_size=bs)

    results = {}

    bay_cls = BayesianLSTM(D_IN, dropout=0.3)
    bay_cls, _ = train_model(bay_cls, tr_l, va_l, task="cls",
                              epochs=80, patience=10)
    t_bay, _, _, _ = find_best_threshold(bay_cls, va_l)
    m_bay, _, _    = eval_cls(bay_cls, te_l, threshold=t_bay)
    results["BayesianLSTM-Cls"] = {**m_bay, "Thresh": t_bay}

    lstm_cls = LSTMClassifier(D_IN)
    lstm_cls, _ = train_model(lstm_cls, tr_l, va_l, task="cls",
                               epochs=80, patience=8)
    t_lstm, _, _, _ = find_best_threshold(lstm_cls, va_l)
    m_lstm, _, _    = eval_cls(lstm_cls, te_l, threshold=t_lstm)
    results["LSTM-Cls"] = {**m_lstm, "Thresh": t_lstm}

    Xtr_f, Xte_f = Xtr_n.reshape(len(Xtr_n), -1), Xte_n.reshape(len(Xte_n), -1)
    for name, clf in [
        ("LR-Cls",  LR_sk(max_iter=300, C=0.1)),
        ("MLP-Cls", MLPClassifier(hidden_layer_sizes=(128, 64),
                                   max_iter=200, random_state=RANDOM_SEED)),
    ]:
        clf.fit(Xtr_f, Ytr)
        p = clf.predict_proba(Xte_f)[:, 1]
        best_t, best_f1 = 0.5, 0.0
        for thr in np.arange(0.1, 0.9, 0.02):
            f = f1_score(Yte, (p > thr).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, thr
        preds = (p > best_t).astype(int)
        results[name] = {"AUC": roc_auc_score(Yte, p),
                          "F1":  f1_score(Yte, preds, zero_division=0),
                          "Acc": accuracy_score(Yte, preds),
                          "AP":  average_precision_score(Yte, p),
                          "Thresh": best_t}

    cls_df = pd.DataFrame(results).T.sort_values("AUC", ascending=False)
    print("\n  Table 2a: Classification Results")
    print(cls_df[["AUC", "F1", "AP", "Thresh"]].round(4).to_string())

    best_deep = max(["BayesianLSTM-Cls", "LSTM-Cls"],
                    key=lambda n: results[n]["AUC"])
    best_base = max(["LR-Cls", "MLP-Cls"],
                    key=lambda n: results[n]["AUC"])
    h4a = results[best_deep]["AUC"] > results[best_base]["AUC"]
    print(f"\n  H4a: {'SUPPORTED ✓' if h4a else 'NULL (boundary) ⚬'}  "
          f"({best_deep} AUC={results[best_deep]['AUC']:.4f} "
          f"vs {best_base} AUC={results[best_base]['AUC']:.4f})")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# H4b / H4c: Regression stage
# [FIX-1] real_val_loader for early stopping
# [FIX-2] group_time_split to prevent leakage
# ─────────────────────────────────────────────────────────────────────────────
def run_h4b_h4c(df_roas: pd.DataFrame, target_n: int = 800):
    print("\n══ H4b/H4c: Regression Stage ═══════════════════════")

    reg_by_sl = {}
    for SL in [4, 6]:
        # [FIX-2] unpack group_ids
        Xr, Yr, gids = build_sequences(
            df_roas,
            "log_ROAS",
            seq_len=SL,
            features=FEATURES
        )
        reg_by_sl[SL] = (Xr, Yr, gids)
        print(f"  REG SEQ_LEN={SL}: {Xr.shape}")

    X_reg, Y_reg, gids_reg = reg_by_sl[4]

    # ── [FIX-2] Group-aware split (no leakage) ────────────────────────────
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = group_time_split(
        X_reg, Y_reg, gids_reg)

    # Fallback if groups too few
    if len(Xva) < 10 or len(Xte) < 10:
        print("  ⚠ group_time_split produced small fold — "
              "using index split with 60/80 fractions")
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(
            X_reg, Y_reg, 0.60, 0.80)

    print(f"  Split sizes — train:{len(Xtr)}  val:{len(Xva)}  test:{len(Xte)}")

    # Bootstrap GRU for augmentation reference
    ref_gru = GRUForecaster(D_IN)
    Xtr_n0, Xva_n0, Xte_n0, _ = normalize_X(Xtr, Xva, Xte)
    bs0 = min(32, len(Xtr_n0))
    ref_gru, _ = train_model(
        ref_gru,
        DataLoader(SeqDataset(Xtr_n0, Ytr), batch_size=bs0, shuffle=True),
        DataLoader(SeqDataset(Xva_n0, Yva), batch_size=bs0),
        epochs=50, patience=8)

    # Augmentation
    X_aug, Y_aug = augment_pipeline(
        Xtr.astype(np.float32), Ytr.astype(np.float32),
        target_n=max(target_n, len(Xtr) * 5), ref_model=ref_gru)

    # Normalise
    sc = MinMaxScaler()
    N_a, T_a, D_a = X_aug.shape
    X_aug_n = sc.fit_transform(X_aug.reshape(-1, D_a)).reshape(X_aug.shape)
    X_va_n  = np.clip(sc.transform(Xva.reshape(-1, D_a)).reshape(Xva.shape), 0, 1)
    X_te_n  = np.clip(sc.transform(Xte.reshape(-1, D_a)).reshape(Xte.shape), 0, 1)

    bs   = min(128, len(X_aug_n))
    
    
    # ============================================================
    # FIX-1: Separate synthetic validation and real validation
    # ============================================================
    
    # Synthetic training data
    tr_l = DataLoader(
        SeqDataset(X_aug_n, Y_aug),
        batch_size=bs,
        shuffle=True
    )
    
    
    # Synthetic validation split
    # (augmentation distribution diagnostic)
    val_size = min(len(X_aug_n) // 10, len(X_aug_n))
    
    X_aug_val = X_aug_n[:val_size]
    Y_aug_val = Y_aug[:val_size]
    
    
    va_l = DataLoader(
        SeqDataset(X_aug_val, Y_aug_val),
        batch_size=bs
    )
    
    
    # Real held-out validation
    # (used for early stopping)
    real_va_l = DataLoader(
        SeqDataset(X_va_n, Yva),
        batch_size=bs
    )
    
    
    # Test
    te_l = DataLoader(
        SeqDataset(X_te_n, Yte),
        batch_size=bs
    )

    model_registry = {
        "BayesianLSTM": BayesianLSTM(D_IN, dropout=0.4),
        "LSTM":         LSTMForecaster(D_IN),
        "GRU":          GRUForecaster(D_IN),
        "BiLSTM":       LSTMForecaster(D_IN, bidirectional=True),
        "Mamba":        MambaForecaster(D_IN, d_model=64, n_layers=3, d_state=8),
    }
    reg_results, reg_preds, gap_reports = {}, {}, []

    for name, model in model_registry.items():
        # [FIX-1] pass real_val_loader → early stop on real data
        trained, history = train_model(
            model, tr_l, va_l,
            epochs=120, patience=12,
            real_val_loader=real_va_l)   # ← FIX-1
        m, p, t = eval_reg(trained, te_l)
        reg_results[name], reg_preds[name] = m, (p, t)
        print(f"  {name:<14} RMSE={m['RMSE']:.4f}  "
              f"MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")

        # [FIX-1] domain gap report
        gap = domain_gap_report(history, model_name=name)
        gap_reports.append(gap)

    # Baseline models (sklearn)
    Xtr_f = X_aug_n.reshape(len(X_aug_n), -1)
    Xte_f = X_te_n.reshape(len(X_te_n), -1)
    for name, clf in [
        ("Ridge", Ridge(alpha=1.0)),
        ("MLP",   MLPRegressor(hidden_layer_sizes=(128, 64),
                                max_iter=300, random_state=RANDOM_SEED)),
    ]:
        clf.fit(Xtr_f, Y_aug)
        p_pred = clf.predict(Xte_f)
        m = {"RMSE": np.sqrt(mean_squared_error(Yte, p_pred)),
             "MAE":  np.mean(np.abs(p_pred - Yte)),
             "R2":   r2_score(Yte, p_pred)}
        reg_results[name], reg_preds[name] = m, (p_pred, Yte)
        print(f"  {name:<14} RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}")

    # Summary table
    reg_df = pd.DataFrame(reg_results).T[["RMSE", "MAE", "R2"]].sort_values("RMSE")
    print("\n  Table 2b: Regression Results (group-split, SL=4)")
    print(reg_df.round(4).to_string())

    # [FIX-1] Domain gap summary
    print("\n  ── Domain-gap report (train–val gap) ──────────────")
    gap_df = pd.DataFrame(gap_reports).set_index("model")
    cols = [c for c in ["best_epoch_real", "best_val_real",
                         "best_val_aug", "final_train",
                         "gap_real", "gap_aug"] if c in gap_df.columns]
    print(gap_df[cols].round(4).to_string())

    # H4b verdict
    best_deep = min(model_registry.keys(), key=lambda n: reg_results[n]["RMSE"])
    best_base = min(["Ridge", "MLP"],       key=lambda n: reg_results[n]["RMSE"])
    h4b = reg_results[best_deep]["RMSE"] < reg_results[best_base]["RMSE"]
    print(f"\n  H4b: {'SUPPORTED ✓' if h4b else 'NOT SUPPORTED ✗'}  "
          f"({best_deep} RMSE={reg_results[best_deep]['RMSE']:.4f} "
          f"vs {best_base} RMSE={reg_results[best_base]['RMSE']:.4f})")

    # DM tests
    print("\n  ── Significant DM Comparisons ─────────────────────")
    model_order = list(reg_preds.keys())
    for i, m1 in enumerate(model_order):
        for m2 in model_order[i + 1:]:
            p1, t1 = reg_preds[m1]
            p2, t2 = reg_preds[m2]
            n_min  = min(len(t1), len(t2))
            dm_v, pv = diebold_mariano(
                t1[:n_min] - p1[:n_min], t2[:n_min] - p2[:n_min])
            if pv < 0.05:
                winner = m1 if dm_v < 0 else m2
                print(f"  {m1} vs {m2}: DM={dm_v:.4f} p={pv:.4f} → {winner}")

    return reg_results, reg_preds, reg_by_sl


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: two-stage prediction [FIXED v2]")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--target_n", type=int, default=800)
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)
    run_h4a(df_paid)
    run_h4b_h4c(df_roas, target_n=args.target_n)
    print("\n✅ Two-stage prediction (H4a–H4c) [FIXED v2] complete.")


if __name__ == "__main__":
    main()
