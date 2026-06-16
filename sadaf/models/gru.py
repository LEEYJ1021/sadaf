"""
sadaf/models/gru.py
GRU-based sequence forecaster for log-ROAS prediction (H4b).

Architecture:
  Input → GRU(hidden=128, layers=2) → LayerNorm → Linear(64)
        → GELU → Dropout → Linear(1)

Used as:
  - Primary baseline RNN in the SADAF regression stage.
  - Reference model for FSD computation in augmentation.
  - Source-domain model in domain-adaptation (H6).
"""

import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    """
    Multi-layer GRU for log-ROAS regression on sparse ad sequences.

    Parameters
    ----------
    input_dim : int
        Number of input features per time-step.
    hidden : int
        GRU hidden state dimension. Default 128.
    layers : int
        Number of stacked GRU layers. Default 2.
    dropout : float
        Dropout probability applied between GRU layers and in head.
        Default 0.2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_dim)

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)
