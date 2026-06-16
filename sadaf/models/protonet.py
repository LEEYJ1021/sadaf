"""
sadaf/models/protonet.py
Prototypical Network encoder for K-shot cold-start ROAS inference.

Motivation (SADAF §6 / RQ4):
  New ads have zero historical ROAS.  ProtoNet infers ROAS from K
  "support" observations (nearby ad groups in embedding space) via
  cosine-similarity-weighted aggregation.

Architecture:
  GRU encoder → LayerNorm → Linear projection (hidden → proj_dim)

K-shot inference:
  For each query, pick K support sequences, compute their prototypes,
  and predict target ROAS as a softmax-weighted average.

Results (SADAF §6):
  K=1 RMSE=2.32, K=5 RMSE=2.28 vs full-data GRU=1.57.
  Converges toward full-data baseline as K grows (W5 supplement).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProtoNetEncoder(nn.Module):
    """
    GRU-based sequence encoder for prototypical learning.

    Parameters
    ----------
    input_dim : int
        Number of input features per time-step.
    hidden : int
        GRU hidden dimension. Default 64.
    proj_dim : int
        Output embedding dimension. Default 32.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        proj_dim: int = 32,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, proj_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_dim)

        Returns
        -------
        torch.Tensor, shape (batch, proj_dim)
        """
        _, h = self.gru(x)
        return self.proj(h[-1])


def kshot_predict(
    encoder: ProtoNetEncoder,
    X_support: np.ndarray,
    Y_support: np.ndarray,
    X_query: np.ndarray,
    temperature: float = 5.0,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    K-shot ROAS prediction for a batch of query sequences.

    Parameters
    ----------
    encoder : ProtoNetEncoder
        Trained encoder (in eval mode).
    X_support : np.ndarray, shape (K, T, D)
        Support sequences.
    Y_support : np.ndarray, shape (K,)
        ROAS labels for support sequences.
    X_query : np.ndarray, shape (Q, T, D)
        Query sequences.
    temperature : float
        Softmax sharpness. Default 5.0.
    device : torch.device or None

    Returns
    -------
    np.ndarray, shape (Q,)
        Predicted log-ROAS for each query.
    """
    if device is None:
        device = next(encoder.parameters()).device

    encoder.eval()
    with torch.no_grad():
        sup_t = torch.FloatTensor(X_support).to(device)
        qry_t = torch.FloatTensor(X_query).to(device)

        sup_emb = encoder(sup_t)              # (K, proj_dim)
        qry_emb = encoder(qry_t)              # (Q, proj_dim)
        proto   = sup_emb.mean(0, keepdim=True)  # (1, proj_dim)

        # Cosine similarity: (Q, K)
        sims = F.cosine_similarity(
            qry_emb.unsqueeze(1),   # (Q, 1, proj_dim)
            sup_emb.unsqueeze(0),   # (1, K, proj_dim)
            dim=-1,
        )
        weights = F.softmax(sims * temperature, dim=-1).cpu().numpy()  # (Q, K)
        preds   = weights @ Y_support  # (Q,)

    return preds
