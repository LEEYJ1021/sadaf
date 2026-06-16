"""
sadaf/models/attention.py
LSTM with additive (Bahdanau-style) temporal attention for log-ROAS regression.

Role in SADAF (§7 / H5):
  Used as the 4th attribution method alongside GS-SHAP, IntGrad, and
  Perm-SHAP.  Attention weights capture *which time-step* is most
  influential — a TEMPORAL axis, complementary to the FEATURE axis
  measured by gradient-based methods (W4 reframing).

Result:
  LSTM-Attn RMSE = 1.3417  R² = 0.6495
  Avg Spearman ρ with GS-SHAP/IntGrad/Perm-SHAP: low (C1, C2)
  → Repositioned as "temporal focus analysis" (not triangulation member).
"""

import torch
import torch.nn as nn


class LSTMWithAttention(nn.Module):
    """
    LSTM with a learned temporal attention mechanism.

    The attention layer computes a scalar score per time-step,
    applies softmax, and uses the weighted sum of hidden states
    as the input to the prediction head.

    Parameters
    ----------
    input_dim : int
        Number of input features per time-step.
    hidden : int
        LSTM hidden dimension. Default 128.
    layers : int
        Number of LSTM layers. Default 2.
    dropout : float
        Dropout applied in LSTM and head. Default 0.2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_dim)
        return_attn : bool
            If True, also return attention weight tensor.

        Returns
        -------
        pred : torch.Tensor, shape (batch,)
        weights : torch.Tensor, shape (batch, seq_len)
            Returned only when return_attn=True.
        """
        out, _ = self.lstm(x)                      # (batch, T, hidden)
        scores  = self.attn(out)                   # (batch, T, 1)
        weights = torch.softmax(scores, dim=1)     # (batch, T, 1)
        context = (weights * out).sum(dim=1)       # (batch, hidden)
        pred    = self.head(context).squeeze(-1)   # (batch,)

        if return_attn:
            return pred, weights.squeeze(-1)       # (batch, T)
        return pred
