"""
05_prediction.py — Two-stage prediction: H4a (classification),
H4b (regression vs. baselines, DM-tested), H4c (Mamba SEQ_LEN robustness).

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
from sklearn.utils import resample

from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import build_sequences, time_split, normalize_X
from sadaf.augmentation.pipeline import augment_pipeline
from sadaf.models.lstm import LSTMForecaster, LSTMClassifier, BayesianLSTM
from sadaf.models.gru import GRUForecaster
from sadaf.models.mamba import MambaForecaster
from sadaf.training.trainer import (train_model, eval_reg, eval_cls,
                                     find_best_threshold, diebold_mariano,
                                     SeqDataset)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = ["CTR", "CVR", "Depth", "log_cost",
            "log_impression", "hour_sin", "hour_cos"]
D_IN = len(FEATURES)


def run_h4a(df_paid: pd.DataFrame):
    print("\n══ H4a: Classification Stage ═══════════════════════")
    X_cls, Y_cls = build_sequences(df_paid, "has_roas", FEATURES, seq_len=4)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(X_cls, Y_cls)
    Xtr_n, Xva_n, Xte_n, _ = normalize_X(Xtr, Xva, Xte)

    bs = min(512, len(Xtr_n))
    tr_l = DataLoader(SeqDataset(Xtr_n, Ytr), batch_size=bs, shuffle=True)
    va_l = DataLoader(SeqDataset(Xva_n, Yva), batch_size=bs)
    te_l = DataLoader(SeqDataset(Xte_n, Yte), batch_size=bs)

    results = {}

    bay_cls = BayesianLSTM(D_IN, dropout=0.3)
    bay_cls, _ = train_model(bay_cls, tr_l, va_l, task="cls", epochs=80, patience=10)
    t_bay, _, _, _ = find_best_threshold(bay_cls, va_l)
    m_bay, _, _ = eval_cls(bay_cls, te_l, threshold=t_bay)
    results["BayesianLSTM-Cls"] = {**m_bay, "Thresh": t_bay}

    lstm_cls = LSTMClassifier(D_IN)
    lstm_cls, _ = train_model(lstm_cls, tr_l, va_l, task="cls", epochs=80, patience=8)
    t_lstm, _, _, _ = find_best_threshold(lstm_cls, va_l)
    m_lstm, _, _ = eval_cls(lstm_cls, te_l, threshold=t_lstm)
    results["LSTM-Cls"] = {**m_lstm, "Thresh": t_lstm}

    Xtr_f, Xte_f = Xtr_n.reshape(len(Xtr_n), -1), Xte_n.reshape(len(Xte_n), -1)
    for name, clf in [("LR-Cls", LR_sk(max_iter=300, C=0.1)),
                       ("MLP-Cls", MLPClassifier(hidden_layer_sizes=(128, 64),
                                                  max_iter=200, random_state=RANDOM_SEED))]:
        clf.fit(Xtr_f, Ytr)
        p = clf.predict_proba(Xte_f)[:, 1]
        best_t, best_f1 = 0.5, 0.0
        for thr in np.arange(0.1, 0.9, 0.02):
            f = f1_score(Yte, (p > thr).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, thr
        preds = (p > best_t).astype(int)
        results[name] = {"AUC": roc_auc_score(Yte, p),
                          "F1": f1_score(Yte, preds, zero_division=0),
                          "Acc": accuracy_score(Yte, preds),
                          "AP": average_precision_score(Yte, p),
                          "Thresh": best_t}

    cls_df = pd.DataFrame(results).T.sort_values("AUC", ascending=False)
    print("\n  Table 2a: Classification Results")
    print(cls_df[["AUC", "F1", "AP", "Thresh"]].round(4).to_string())

    best_deep = max(["BayesianLSTM-Cls", "LSTM-Cls"], key=lambda n: results[n]["AUC"])
    best_base = max(["LR-Cls", "MLP-Cls"], key=lambda n: results[n]["AUC"])
    h4a_supported = results[best_deep]["AUC"] > results[best_base]["AUC"]
    print(f"\n  H4a: {'SUPPORTED ✓' if h4a_supported else 'NULL (boundary) ⚬'}  "
          f"({best_deep} AUC={results[best_deep]['AUC']:.4f} "
          f"vs {best_base} AUC={results[best_base]['AUC']:.4f})")
    return results


def run_h4b_h4c(df_roas: pd.DataFrame, target_n: int = 800):
    print("\n══ H4b/H4c: Regression Stage ═══════════════════════")
    reg_by_sl = {}
    for SL in [4, 6]:
        Xr, Yr = build_sequences(df_roas, "log_ROAS", FEATURES, seq_len=SL)
        reg_by_sl[SL] = (Xr, Yr)
        print(f"  REG SEQ_LEN={SL}: {Xr.shape}")

    X_reg, Y_reg = reg_by_sl[4]
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(X_reg, Y_reg)
    if len(Xva) < 10 or len(Xte) < 10:
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = time_split(X_reg, Y_reg, 0.60, 0.80)

    ref_gru = GRUForecaster(D_IN)
    Xtr_n0, Xva_n0, Xte_n0, _ = normalize_X(Xtr, Xva, Xte)
    bs0 = min(32, len(Xtr_n0))
    ref_gru, _ = train_model(
        ref_gru,
        DataLoader(SeqDataset(Xtr_n0, Ytr), batch_size=bs0, shuffle=True),
        DataLoader(SeqDataset(Xva_n0, Yva), batch_size=bs0),
        epochs=50, patience=8)

    X_aug, Y_aug = augment_pipeline(
        Xtr.astype(np.float32), Ytr.astype(np.float32),
        target_n=max(target_n, len(Xtr) * 5), ref_lstm=ref_gru)

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    N_a, T_a, D_a = X_aug.shape
    X_aug_n = sc.fit_transform(X_aug.reshape(-1, D_a)).reshape(X_aug.shape)
    X_va_n = np.clip(sc.transform(Xva.reshape(-1, D_a)).reshape(Xva.shape), 0, 1)
    X_te_n = np.clip(sc.transform(Xte.reshape(-1, D_a)).reshape(Xte.shape), 0, 1)

    bs = min(128, len(X_aug_n))
    tr_l = DataLoader(SeqDataset(X_aug_n, Y_aug), batch_size=bs, shuffle=True)
    va_l = DataLoader(SeqDataset(X_va_n, Yva), batch_size=bs)
    te_l = DataLoader(SeqDataset(X_te_n, Yte), batch_size=bs)

    model_registry = {
        "BayesianLSTM": BayesianLSTM(D_IN, dropout=0.4),
        "LSTM": LSTMForecaster(D_IN),
        "GRU": GRUForecaster(D_IN),
        "BiLSTM": LSTMForecaster(D_IN, bidirectional=True),
        "Mamba": MambaForecaster(D_IN, d_model=64, n_layers=3, d_state=8),
    }
    reg_results, reg_preds = {}, {}
    for name, model in model_registry.items():
        trained, _ = train_model(model, tr_l, va_l, epochs=120, patience=12)
        m, p, t = eval_reg(trained, te_l)
        reg_results[name], reg_preds[name] = m, (p, t)
        print(f"  {name:<14} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R²={m['R2']:.4f}")

    Xtr_f, Xte_f = X_aug_n.reshape(len(X_aug_n), -1), X_te_n.reshape(len(X_te_n), -1)
    for name, clf in [("Ridge", Ridge(alpha=1.0)),
                       ("MLP", MLPRegressor(hidden_layer_sizes=(128, 64),
                                             max_iter=300, random_state=RANDOM_SEED))]:
        clf.fit(Xtr_f, Y_aug)
        p_pred = clf.predict(Xte_f)
        m = {"RMSE": np.sqrt(mean_squared_error(Yte, p_pred)),
             "MAE": np.mean(np.abs(p_pred - Yte)),
             "R2": r2_score(Yte, p_pred)}
        reg_results[name], reg_preds[name] = m, (p_pred, Yte)
        print(f"  {name:<14} RMSE={m['RMSE']:.4f} R²={m['R2']:.4f}")

    reg_df = pd.DataFrame(reg_results).T[["RMSE", "MAE", "R2"]].sort_values("RMSE")
    print("\n  Table 2b: Regression Results (augmented, SL=4)")
    print(reg_df.round(4).to_string())

    best_deep = min(model_registry.keys(), key=lambda n: reg_results[n]["RMSE"])
    best_base = min(["Ridge", "MLP"], key=lambda n: reg_results[n]["RMSE"])
    h4b_supported = reg_results[best_deep]["RMSE"] < reg_results[best_base]["RMSE"]
    print(f"\n  H4b: {'SUPPORTED ✓' if h4b_supported else 'NOT SUPPORTED ✗'}  "
          f"({best_deep} RMSE={reg_results[best_deep]['RMSE']:.4f} "
          f"vs {best_base} RMSE={reg_results[best_base]['RMSE']:.4f})")

    print("\n  ── Significant DM Comparisons ─────────────────")
    model_order = list(reg_preds.keys())
    for i, m1 in enumerate(model_order):
        for m2 in model_order[i + 1:]:
            p1, t1 = reg_preds[m1]
            p2, t2 = reg_preds[m2]
            n_min = min(len(t1), len(t2))
            dm_v, pv = diebold_mariano(t1[:n_min] - p1[:n_min], t2[:n_min] - p2[:n_min])
            if pv < 0.05:
                winner = m1 if dm_v < 0 else m2
                print(f"  {m1} vs {m2}: DM={dm_v:.4f} p={pv:.4f} → {winner} better")

    return reg_results, reg_preds, reg_by_sl


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: two-stage prediction (H4a-H4c)")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--target_n", type=int, default=800)
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)
    run_h4a(df_paid)
    run_h4b_h4c(df_roas, target_n=args.target_n)
    print("\n✅ Two-stage prediction (H4a–H4c) complete.")


if __name__ == "__main__":
    main()