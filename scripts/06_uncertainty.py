"""
06_uncertainty.py — Bayesian posterior calibration (MC Dropout + temperature
scaling) and ProtoNet K-shot cold-start ROAS inference.

Usage:
    python scripts/06_uncertainty.py --data_path data/3월성과데이터(샘플).xlsx
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sadaf.data.loader import load_and_preprocess
from sadaf.data.sequence import build_sequences, time_split, normalize_X
from sadaf.augmentation.pipeline import augment_pipeline
from sadaf.models.lstm import BayesianLSTM
from sadaf.models.gru import GRUForecaster
from sadaf.models.protonet import ProtoNetEncoder
from sadaf.training.trainer import train_model, eval_reg, SeqDataset

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = ["CTR", "CVR", "Depth", "log_cost",
            "log_impression", "hour_sin", "hour_cos"]
D_IN = len(FEATURES)


def run_bayesian_uncertainty(X_aug_n, Y_aug, X_te_n, Y_te, tr_l, va_l):
    print("\n══ Bayesian Uncertainty Quantification ═════════════")
    bay_reg = BayesianLSTM(D_IN, dropout=0.4)
    bay_reg, _ = train_model(bay_reg, tr_l, va_l, epochs=120, patience=12)

    posterior = bay_reg.predict_posterior(X_te_n, n_samples=500, temperature=1.5)

    print("\n  Bayesian Calibration Report:")
    draws = posterior["draws"]
    for alpha in [0.50, 0.80, 0.90, 0.95]:
        lo_ = np.percentile(draws, (1 - alpha) / 2 * 100, axis=0)
        hi_ = np.percentile(draws, (1 - (1 - alpha) / 2) * 100, axis=0)
        cov_ = np.mean((Y_te >= lo_) & (Y_te <= hi_)) * 100
        status = "✓" if cov_ >= alpha * 100 - 3 else "⚠"
        print(f"  {int(alpha * 100):3d}% nominal → {cov_:5.1f}% actual  {status}")

    coverage_95 = np.mean(
        (Y_te >= posterior["ci_lo"]) & (Y_te <= posterior["ci_hi"])) * 100
    mean_width = np.mean(posterior["ci_hi"] - posterior["ci_lo"])
    print(f"\n  95% CI coverage : {coverage_95:.1f}%")
    print(f"  Mean width      : {mean_width:.4f}")
    print(f"  Temperature     : {posterior['temperature']}")
    return bay_reg, posterior


def run_protonet(X_aug_n, Y_aug, X_te_n, Y_te, gru_full_rmse):
    print("\n══ ProtoNet: K-Shot Cold-Start ROAS Inference ══════")
    proto_encoder = ProtoNetEncoder(D_IN, hidden=64, proj_dim=32).to(DEVICE)
    proto_opt = torch.optim.Adam(proto_encoder.parameters(), lr=1e-3)
    proto_encoder.train()

    X_t = torch.FloatTensor(X_aug_n).to(DEVICE)
    bins = pd.qcut(Y_aug, q=3, labels=False, duplicates="drop")
    for ep in range(100):
        proto_opt.zero_grad()
        embs = proto_encoder(X_t)
        group_loss = torch.tensor(0.0).to(DEVICE)
        for g in np.unique(bins):
            mask = torch.tensor(bins == g).to(DEVICE)
            if mask.sum() < 2:
                continue
            g_emb = embs[mask]
            proto = g_emb.mean(0, keepdim=True)
            group_loss = group_loss + ((g_emb - proto) ** 2).sum(-1).mean()
        group_loss.backward()
        proto_opt.step()
        if (ep + 1) % 50 == 0:
            print(f"    ProtoNet Ep {ep + 1}: within-cluster loss={group_loss.item():.4f}")

    print("\n  K-shot ROAS prediction:")
    print(f"  {'K':>4}  {'RMSE':>8}  {'n_eval':>8}")
    proto_encoder.eval()
    kshot_results = {}
    for k in [1, 2, 3, 5]:
        rmses = []
        with torch.no_grad():
            for i in range(k, len(X_te_n)):
                sup_X = torch.FloatTensor(X_te_n[i - k:i]).to(DEVICE)
                qry_X = torch.FloatTensor(X_te_n[i:i + 1]).to(DEVICE)
                sup_e = proto_encoder(sup_X)
                qry_e = proto_encoder(qry_X)[0]
                sup_Y = Y_te[i - k:i]
                sims = F.cosine_similarity(qry_e.unsqueeze(0), sup_e, dim=-1)
                weights = F.softmax(sims * 5, dim=0).cpu().numpy()
                pred = (weights * sup_Y).sum()
                rmses.append((pred - Y_te[i]) ** 2)
        rmse_k = np.sqrt(np.mean(rmses)) if rmses else np.nan
        kshot_results[k] = rmse_k
        print(f"  {k:>4}  {rmse_k:>8.4f}  {len(rmses):>8}")
    print(f"\n  Baseline GRU (full data): RMSE={gru_full_rmse:.4f}")
    return proto_encoder, kshot_results


def main():
    parser = argparse.ArgumentParser(
        description="SADAF: Bayesian uncertainty + ProtoNet")
    parser.add_argument("--data_path", type=str,
                         default="data/3월성과데이터(샘플).xlsx")
    parser.add_argument("--target_n", type=int, default=800)
    args = parser.parse_args()

    df, df_paid, df_roas = load_and_preprocess(args.data_path)
    X_reg, Y_reg = build_sequences(df_roas, "log_ROAS", FEATURES, seq_len=4)
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
    gru_metrics, _, _ = eval_reg(
        ref_gru, DataLoader(SeqDataset(Xte_n0, Yte), batch_size=bs0))

    X_aug, Y_aug = augment_pipeline(
        Xtr.astype(np.float32), Ytr.astype(np.float32),
        target_n=max(args.target_n, len(Xtr) * 5), ref_lstm=ref_gru)

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    N_a, T_a, D_a = X_aug.shape
    X_aug_n = sc.fit_transform(X_aug.reshape(-1, D_a)).reshape(X_aug.shape)
    X_va_n = np.clip(sc.transform(Xva.reshape(-1, D_a)).reshape(Xva.shape), 0, 1)
    X_te_n = np.clip(sc.transform(Xte.reshape(-1, D_a)).reshape(Xte.shape), 0, 1)

    bs = min(128, len(X_aug_n))
    tr_l = DataLoader(SeqDataset(X_aug_n, Y_aug), batch_size=bs, shuffle=True)
    va_l = DataLoader(SeqDataset(X_va_n, Yva), batch_size=bs)

    run_bayesian_uncertainty(X_aug_n, Y_aug, X_te_n, Yte, tr_l, va_l)
    run_protonet(X_aug_n, Y_aug, X_te_n, Yte, gru_metrics["RMSE"])
    print("\n✅ Bayesian uncertainty + ProtoNet complete.")


if __name__ == "__main__":
    main()