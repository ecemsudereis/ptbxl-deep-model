from __future__ import annotations
import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, in_ch: int = 12, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_ch = in_ch

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

        self.enc_mu = nn.Linear(128, latent_dim)
        self.enc_logvar = nn.Linear(128, latent_dim)

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

    def encode(self, x: torch.Tensor):
        h = self.encoder_conv(x)
        h = self.encoder_pool(h).squeeze(-1)
        return self.enc_mu(h), self.enc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.size(0), 128, 125)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


if __name__ == "__main__":
    m = VAE()
    x = torch.randn(4, 12, 1000)
    r, mu, lv = m(x)
    print("recon:", tuple(r.shape), "| mu:", tuple(mu.shape),
          "| KL:", float(m.kl_loss(mu, lv).detach()))
    assert r.shape == (4, 12, 1000) and mu.shape == (4, 64)
    print("OK — VAE shape contract holds.")
