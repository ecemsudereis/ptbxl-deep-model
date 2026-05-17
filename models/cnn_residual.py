"""Block 2+3 — 1D CNN + Residual (Person B).

COURSE JUSTIFICATION:
    CNN (W6): ECG diagnosis relies on LOCAL patterns such as P-QRS-T
    wave morphology. Convolution's sparse connectivity, parameter
    sharing, and translation equivariance (W6) make it ideal for
    capturing this morphology anywhere in the signal.

    Residual (W6 ResNet + W5): In deep nets the vanishing gradient
    problem (W5) is mitigated by skip connections — He et al. 2016.
    Residual blocks can be toggled as a DISTINCT block in the ablation
    (use_residual flag).

Contract (CONTRACT.md):
    forward(x: [B,12,1000]) -> [B, C, L']   (C channels, L' shortened time)
    L' feeds the time axis of the recurrent block.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """A single 1D residual block: Conv-BN-ReLU ×2 + skip."""

    def __init__(self, ch: int, kernel: int = 7):
        super().__init__()
        # TODO(B): two Conv1d + BatchNorm1d + ReLU, add input to output.
        self.body = nn.Identity()  # placeholder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)  # skip connection (residual)


class CNNResidual(nn.Module):
    """1D CNN backbone + optional residual blocks.

    Args:
        in_ch: input channels (12).
        base_ch: first conv channel count.
        n_blocks: number of residual blocks.
        use_residual: if False, skip connections disabled (ablation).
    """

    def __init__(self, in_ch: int = 12, base_ch: int = 64,
                 n_blocks: int = 4, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.out_channels = base_ch
        # TODO(B): stem Conv1d (in_ch->base_ch) + downsampling +
        #          n_blocks ResidualBlock1D (fall back to plain Conv
        #          blocks if use_residual=False).
        self.net = nn.Conv1d(in_ch, base_ch, kernel_size=7,
                             stride=4, padding=3)  # placeholder stem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, out_channels, L']."""
        return self.net(x)


if __name__ == "__main__":
    m = CNNResidual()
    x = torch.randn(4, 12, 1000)
    y = m(x)
    print("CNN output:", tuple(y.shape), "(B, C, L')")
    assert y.shape[0] == 4 and y.shape[1] == m.out_channels
    print("OK — CNN+Residual shape contract holds.")
