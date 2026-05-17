"""Block 1 — Denoising Autoencoder (Person B).

COURSE JUSTIFICATION (W8):
    ECG signals are inherently noisy (baseline wander, muscle artifact,
    powerline). The denoising AE covered in W8 is forced to reconstruct
    the clean input from a deliberately corrupted version, so it learns
    the data manifold (Vincent et al. 2008; Goodfellow Ch. 14.5). This
    avoids the trivial identity trap and gives the classifier a
    noise-robust representation. Used as unsupervised pretraining.

Contract (CONTRACT.md):
    forward(x: [B,12,1000]) -> (recon: [B,12,1000], latent: [B,D_dae])
    Training: input is corrupted, clean x is the target (MSE).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """1D convolutional denoising autoencoder.

    Args:
        in_ch: number of leads (12).
        latent_dim: code size (undercomplete bottleneck).
        noise_std: std of the Gaussian noise added in training (corruption).
    """

    def __init__(self, in_ch: int = 12, latent_dim: int = 128,
                 noise_std: float = 0.1):
        super().__init__()
        self.noise_std = noise_std
        self.latent_dim = latent_dim
        # TODO(B): Encoder (Conv1d stack → bottleneck latent_dim)
        # TODO(B): Decoder (ConvTranspose1d → (B,12,1000) reconstruction)
        # Temporary placeholder (so the smoke test shapes hold):
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """Corrupt input during training (Gaussian noise). Identity at test."""
        if self.training:
            return x + self.noise_std * torch.randn_like(x)
        return x

    def forward(self, x: torch.Tensor):
        """Returns (recon [B,12,1000], latent [B,latent_dim])."""
        # TODO(B): real encode/decode. For now a shape-compatible stub:
        latent = torch.zeros(x.size(0), self.latent_dim, device=x.device)
        recon = x  # placeholder
        return recon, latent

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Denoising objective: reconstruct clean x from corrupted input (MSE)."""
        recon, _ = self.forward(self.corrupt(x))
        return nn.functional.mse_loss(recon, x)


if __name__ == "__main__":
    m = DenoisingAutoencoder()
    x = torch.randn(4, 12, 1000)
    r, z = m(x)
    print("recon:", tuple(r.shape), "| latent:", tuple(z.shape))
    assert r.shape == (4, 12, 1000) and z.shape == (4, 128)
    print("OK — DAE shape contract holds.")
