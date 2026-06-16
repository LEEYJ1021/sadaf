"""
sadaf/augmentation/vae.py
--------------------------
β-VAE architecture for sequence augmentation.

The encoder is an LSTM that summarises an input sequence into a latent
distribution (μ, log σ²).  The decoder is an MLP that maps a latent vector
back to a full sequence.  The β weighting controls the KL divergence penalty:
  β > 1 encourages more disentangled representations.

Paper context: β = 1.0 (standard VAE) was used in the SADAF experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sadaf.config import VAE_LATENT_DIM, VAE_BETA, DEVICE


class AdSequenceVAE(nn.Module):
    """
    Variational Autoencoder for short ad-performance sequences.

    Encoder : 1-layer LSTM → (μ, log σ²) projection
    Decoder : Linear + GELU → reshape to (seq_len, input_dim)

    Parameters
    ----------
    seq_len   : int   — number of time steps (T)
    input_dim : int   — number of features per time step (D)
    latent_dim: int   — dimension of latent space z
    beta      : float — KL weight; 1.0 = standard VAE
    """

    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        latent_dim: int = VAE_LATENT_DIM,
        beta: float = VAE_BETA,
    ):
        super().__init__()
        self.beta      = beta
        self.seq_len   = seq_len
        self.input_dim = input_dim

        # Encoder
        self.enc_lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.mu_proj  = nn.Linear(64, latent_dim)
        self.lv_proj  = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, seq_len * input_dim),
        )

    def encode(self, x: torch.Tensor):
        _, (h, _) = self.enc_lstm(x)
        h = h[-1]
        return self.mu_proj(h), self.lv_proj(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, self.seq_len, self.input_dim)

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        x_recon : Tensor, shape (B, seq_len, input_dim)
        mu      : Tensor, shape (B, latent_dim)
        logvar  : Tensor, shape (B, latent_dim)
        loss    : Tensor scalar — ELBO = reconstruction + β × KL
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decode(z)

        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        kld        = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        return x_recon, mu, logvar, recon_loss + self.beta * kld
