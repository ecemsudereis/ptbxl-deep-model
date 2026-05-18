from __future__ import annotations
import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):

    def __init__(self, in_ch: int = 12, latent_dim: int = 128,
                 noise_std: float = 0.1):
        super().__init__()
        self.noise_std = noise_std
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.encoder_pool = nn.AdaptiveAvgPool1d(1)
        self.encoder_fc = nn.Linear(128, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 128 * 125)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, in_ch, kernel_size=7, stride=2,
                               padding=3, output_padding=1),
        )

    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x + self.noise_std * torch.randn_like(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h = self.encoder_pool(h).squeeze(-1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.size(0), 128, 125)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(self.corrupt(x))
        return nn.functional.mse_loss(recon, x)


if __name__ == "__main__":
    m = DenoisingAutoencoder()
    x = torch.randn(4, 12, 1000)
    r, z = m(x)
    print("recon:", tuple(r.shape), "| latent:", tuple(z.shape))
    assert r.shape == (4, 12, 1000) and z.shape == (4, 128)
    loss = m.reconstruction_loss(x)
    print("recon loss:", float(loss.detach()))
    print("OK — DAE shape contract holds.")
