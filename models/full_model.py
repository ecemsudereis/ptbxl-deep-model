"""Main Model — coherent fusion of the 6 blocks (Person B).

DATA FLOW (coherent — the requirement the instructor stressed twice):

    Raw ECG  [B,12,1000]
        │
        ▼  (1) Denoising AE  ── denoised representation, unsupervised
        │                       pretraining (W8)
    denoised [B,12,1000]
        │
        ▼  (2+3) 1D CNN + Residual ── local P-QRS-T morphology (W6+W5)
    [B, C, L']
        │
        ▼  (4) BiLSTM/BiGRU ── temporal/rhythm dependency (W7)
    [B, L', H]
        │
        ▼  (5) Attention ── focus on diagnostic segments +
        │                    interpretability (W7/W8)
    [B, H]
        │
        ▼  classifier head
    [B, 5]  logits

    (6) VAE ── parallel branch: latent regularization + minority-class
              augmentation. KL term added to the total loss (W8).

ABLATION: the __init__ flags are exactly the axes of the ablation
matrix (see CONTRACT §5, ablation/run_ablation.py).
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .dae import DenoisingAutoencoder
from .cnn_residual import CNNResidual
from .recurrent import RecurrentBlock
from .attention import TemporalAttention, MeanPool
from .vae import VAE

N_CLASSES = 5


class FullModel(nn.Module):
    """End-to-end model. Flags toggle a block on/off for ablation.

    Args:
        use_dae: Denoising AE branch/pretraining on.
        use_residual: residual skips in the CNN on.
        rnn_type: "lstm" | "gru".
        bidirectional: recurrent bidirectional or not.
        use_attention: attention pooling (False → mean pool).
        use_vae: VAE branch + KL regularization on.
    """

    def __init__(self, use_dae: bool = True, use_residual: bool = True,
                 rnn_type: str = "lstm", bidirectional: bool = True,
                 use_attention: bool = True, use_vae: bool = True,
                 cnn_base_ch: int = 64, rnn_hidden: int = 128):
        super().__init__()
        self.flags = dict(use_dae=use_dae, use_residual=use_residual,
                          rnn_type=rnn_type, bidirectional=bidirectional,
                          use_attention=use_attention, use_vae=use_vae)

        self.dae = DenoisingAutoencoder() if use_dae else None
        self.vae = VAE() if use_vae else None

        self.cnn = CNNResidual(in_ch=12, base_ch=cnn_base_ch,
                               use_residual=use_residual)
        self.rnn = RecurrentBlock(in_features=self.cnn.out_channels,
                                  hidden=rnn_hidden, rnn_type=rnn_type,
                                  bidirectional=bidirectional)
        self.pool = (TemporalAttention(self.rnn.out_features)
                     if use_attention else MeanPool())
        self.head = nn.Sequential(
            nn.Linear(self.rnn.out_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                 # W4 regularization
            nn.Linear(128, N_CLASSES),       # logits — sigmoid in the loss
        )

    def forward(self, x: torch.Tensor):
        """x: [B,12,1000] -> (logits [B,5], aux dict)."""
        aux = {}

        if self.dae is not None:
            denoised, _ = self.dae(x)
        else:
            denoised = x

        if self.vae is not None:
            _, mu, logvar = self.vae(x)
            aux["vae"] = (mu, logvar)

        feat = self.cnn(denoised)            # [B, C, L']
        seq = self.rnn(feat)                 # [B, L', H]
        pooled, w = self.pool(seq)           # [B, H], [B,L'] | None
        logits = self.head(pooled)           # [B, 5]

        aux["attn_weights"] = w
        return logits, aux


if __name__ == "__main__":
    # Full model + a few ablation variants shape test.
    x = torch.randn(4, 12, 1000)
    for cfg in [
        dict(),  # full model (6 blocks)
        dict(use_dae=False),
        dict(use_residual=False),
        dict(rnn_type="gru", bidirectional=False),
        dict(use_attention=False),
        dict(use_vae=False),
    ]:
        m = FullModel(**cfg)
        logits, aux = m(x)
        assert logits.shape == (4, 5), f"VIOLATION: {cfg}"
        label = str(cfg) if cfg else "FULL MODEL (6 blocks)"
        print(f"{label:<42} -> logits {tuple(logits.shape)} OK")
    print("OK — all ablation variants hold the shape contract.")
