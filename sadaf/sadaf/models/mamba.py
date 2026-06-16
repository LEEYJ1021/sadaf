"""
sadaf/models/mamba.py
---------------------
Simplified Mamba-style selective state-space model (Gu et al., 2023).

The key inductive bias is an input-dependent Δ_t gate that selectively
compresses zero-ROAS time slots, providing robustness to sequence-length
variation (tested in H4c).

Note: Mamba is NOT claimed to be the most accurate model in SADAF.
      Its primary contribution is ROBUSTNESS (lower ΔRMSE per +2 SEQ_LEN
      compared to LSTM and GRU).  BayesianLSTM achieves the best raw RMSE.

Classes
-------
SelectiveSSM    — Core S6 recurrence with input-dependent Δ, B, C matrices
MambaBlock      — One Mamba block: norm → input projection → conv → SSM → gate
MambaForecaster — Full model: projection + N MambaBlocks + regression head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sadaf.config import MAMBA_D_MODEL, MAMBA_N_LAYERS, MAMBA_D_STATE, MAMBA_DROPOUT


class SelectiveSSM(nn.Module):
    """
    Simplified selective state-space recurrence (S6).

    The state transition matrix A is parameterised as a log-diagonal
    (ensures stability via negative exponential).  B and C matrices are
    projected from the input at each time step, enabling the model to
    selectively gate information based on content.

    Parameters
    ----------
    d_model : int
        Channel / model dimension.
    d_state : int
        State dimension (rank of the hidden state).
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_state = d_state
        # Stable negative diagonal: A = -exp(A_log)
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float)
                .unsqueeze(0)
                .expand(d_model, -1)
            )
        )
        # Skip connection scaling
        self.D = nn.Parameter(torch.ones(d_model))
        # Input-dependent Δ, B, C projection
        self.x_proj  = nn.Linear(d_model, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, L, D)

        Returns
        -------
        y : Tensor, shape (B, L, D)
        """
        B, L, D = x.shape
        A = -torch.exp(self.A_log)                      # (D, d_state)
        xBC = self.x_proj(x)                            # (B, L, d_state*2+1)
        dt_raw, B_mat, C_mat = xBC.split(
            [1, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))           # (B, L, D)

        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            dA = torch.exp(
                dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0)
            )                                           # (B, D, d_state)
            dB = (
                dt[:, t, :].unsqueeze(-1)
                * B_mat[:, t, :].unsqueeze(1)
            )                                           # (B, D, d_state)
            h  = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y  = (h * C_mat[:, t, :].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y)

        out = torch.stack(ys, dim=1)                    # (B, L, D)
        return out + x * self.D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    """
    One Mamba block:
      LayerNorm → input projection (expand × 2) →
      depthwise conv1d → SiLU → SelectiveSSM → gated output → residual

    Parameters
    ----------
    d_model : int
    d_state : int
    expand : int
        Expansion factor for inner dimension.
    dropout : float
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_inner       = d_model * expand
        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, 3, padding=1, groups=d_inner
        )
        self.ssm      = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x        = self.norm(x)
        xz       = self.in_proj(x)              # (B, L, 2*d_inner)
        x_, z    = xz.chunk(2, dim=-1)
        x_       = F.silu(
            self.conv1d(x_.transpose(1, 2)).transpose(1, 2)
        )
        y = self.ssm(x_) * F.silu(z)
        return self.drop(self.out_proj(y)) + residual


class MambaForecaster(nn.Module):
    """
    Full Mamba model for log-ROAS regression.

    Architecture:
        Linear input projection →
        N × MambaBlock →
        LayerNorm →
        2-layer regression head (last time step)

    Parameters
    ----------
    input_dim : int
    d_model : int
        Internal model dimension.
    n_layers : int
        Number of stacked MambaBlocks.
    d_state : int
        SelectiveSSM state dimension.
    dropout : float
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = MAMBA_D_MODEL,
        n_layers: int = MAMBA_N_LAYERS,
        d_state: int = MAMBA_D_STATE,
        dropout: float = MAMBA_DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_state, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h[:, -1, :])).squeeze(-1)
