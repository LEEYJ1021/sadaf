"""
sadaf/models/lstm.py
--------------------
LSTM-based forecasting and classification models used in SADAF.

Classes
-------
LSTMForecaster       — Standard LSTM for log-ROAS regression (H4b)
LSTMClassifier       — Standard LSTM for binary ROAS classification (H4a)
BiLSTMForecaster     — Bidirectional LSTM (robustness baseline)
BayesianLSTM         — MC-Dropout approximate Bayesian LSTM with
                       temperature-scaled posterior (RQ4 / Bayesian UQ)
LSTMWithAttention    — LSTM + scaled dot-product attention (H5 attribution)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from sadaf.config import (
    BAYESIAN_LSTM_DROPOUT,
    BAYESIAN_TEMPERATURE,
    BAYESIAN_MC_SAMPLES,
    DEVICE,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_DROPOUT,
)


# ── Standard LSTM forecaster ───────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    """
    Two-layer LSTM followed by a two-layer MLP head for regression.

    Parameters
    ----------
    input_dim : int
        Number of input features (D_IN = 7 in SADAF).
    hidden : int
        LSTM hidden size.
    layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied between LSTM layers.
    bidirectional : bool
        If True, creates a BiLSTM (doubles effective hidden size).
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = LSTM_HIDDEN,
        layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, layers,
            batch_first=True, dropout=dropout,
            bidirectional=bidirectional,
        )
        mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * mult),
            nn.Linear(hidden * mult, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── BiLSTM alias ──────────────────────────────────────────────────────────────
def BiLSTMForecaster(input_dim: int, **kwargs) -> LSTMForecaster:
    """Convenience constructor for bidirectional LSTM."""
    return LSTMForecaster(input_dim, bidirectional=True, **kwargs)


# ── LSTM classifier (binary) ──────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """
    LSTM for binary has_roas classification (Stage 1 of the two-stage pipeline).
    Outputs raw logits; apply sigmoid externally.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = LSTM_HIDDEN,
        layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, layers,
            batch_first=True, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── Bayesian LSTM (MC Dropout) ─────────────────────────────────────────────────
class BayesianLSTM(nn.Module):
    """
    Approximate Bayesian LSTM via MC Dropout (Gal & Ghahramani, 2016).

    Calibration notes (from diagnostic analysis):
      - Dropout = 0.4, Temperature = 1.5 → 95% CI coverage ≈ 94.1%
      - Keeps model in train() mode during inference to enable stochastic
        forward passes.

    Parameters
    ----------
    input_dim : int
    hidden : int
    layers : int
    dropout : float
        Applied to both LSTM layers and the MLP head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = LSTM_HIDDEN,
        layers: int = LSTM_LAYERS,
        dropout: float = BAYESIAN_LSTM_DROPOUT,
    ):
        super().__init__()
        self.dropout_rate = dropout
        self.lstm = nn.LSTM(
            input_dim, hidden, layers,
            batch_first=True, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

    def predict_posterior(
        self,
        X_np: np.ndarray,
        n_samples: int = BAYESIAN_MC_SAMPLES,
        temperature: float = BAYESIAN_TEMPERATURE,
    ) -> Dict[str, np.ndarray]:
        """
        Generate approximate posterior predictive distribution via MC Dropout.

        The model is kept in train() mode so that dropout masks are sampled
        independently for each forward pass.  Temperature scaling expands the
        spread of the posterior draws to improve empirical coverage.

        Parameters
        ----------
        X_np : np.ndarray, shape (N, seq_len, D)
            Normalised test sequences.
        n_samples : int
            Number of stochastic forward passes.
        temperature : float
            Temperature scalar > 1 widens intervals to correct undercoverage.

        Returns
        -------
        dict with keys:
            mean     — posterior mean, shape (N,)
            std      — posterior std (after temperature scaling), shape (N,)
            ci_lo    — 2.5th percentile of scaled draws, shape (N,)
            ci_hi    — 97.5th percentile, shape (N,)
            draws    — all scaled draws, shape (n_samples, N)
            temperature — value used
        """
        self.train()   # keep dropout active
        X_t = torch.FloatTensor(X_np).to(next(self.parameters()).device)
        draws = []
        with torch.no_grad():
            for _ in range(n_samples):
                draws.append(self(X_t).cpu().numpy())
        draws = np.stack(draws, axis=0)          # (n_samples, N)

        mean       = draws.mean(axis=0)
        deviations = draws - mean
        draws_sc   = mean + deviations * temperature

        std   = draws_sc.std(axis=0)
        ci_lo = np.percentile(draws_sc, 2.5,  axis=0)
        ci_hi = np.percentile(draws_sc, 97.5, axis=0)

        return {
            "mean": mean,
            "std": std,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "draws": draws_sc,
            "temperature": temperature,
        }

    def calibration_report(
        self,
        X_np: np.ndarray,
        Y_np: np.ndarray,
        n_samples: int = BAYESIAN_MC_SAMPLES,
        temperature: float = BAYESIAN_TEMPERATURE,
    ) -> None:
        """Print empirical coverage at 50 / 80 / 90 / 95% nominal levels."""
        posterior = self.predict_posterior(X_np, n_samples, temperature)
        draws_sc  = posterior["draws"]
        print("\n  Bayesian Calibration Report:")
        for alpha in [0.50, 0.80, 0.90, 0.95]:
            lo = np.percentile(draws_sc, (1 - alpha) / 2 * 100, axis=0)
            hi = np.percentile(draws_sc, (1 - (1 - alpha) / 2) * 100, axis=0)
            cov = np.mean((Y_np >= lo) & (Y_np <= hi)) * 100
            status = "✓" if cov >= alpha * 100 - 3 else "⚠"
            print(f"  {int(alpha * 100):3d}% nominal → {cov:5.1f}% actual  {status}")


# ── LSTM with attention ────────────────────────────────────────────────────────
class LSTMWithAttention(nn.Module):
    """
    LSTM with a learned scalar attention over time steps.

    Used as the 4th attribution method (temporal position focus) in H5.
    The attention weights indicate *which time step* the model relied on,
    complementing feature-level importance from GS-SHAP / IntGrad / Perm-SHAP.

    Parameters
    ----------
    input_dim : int
    hidden : int
    layers : int
    dropout : float
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = LSTM_HIDDEN,
        layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, layers,
            batch_first=True, dropout=dropout,
        )
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
        return_attn : bool
            If True, also return attention weights (B, T).

        Returns
        -------
        pred : Tensor, shape (B,)
        weights : Tensor, shape (B, T)   — only if return_attn=True
        """
        out, _   = self.lstm(x)                    # (B, T, H)
        scores   = self.attn(out)                  # (B, T, 1)
        weights  = torch.softmax(scores, dim=1)    # (B, T, 1)
        context  = (weights * out).sum(dim=1)      # (B, H)
        pred     = self.head(context).squeeze(-1)  # (B,)
        if return_attn:
            return pred, weights.squeeze(-1)
        return pred
