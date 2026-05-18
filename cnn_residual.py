from __future__ import annotations
import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):

    def __init__(self, ch: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.body(x))


class PlainBlock1D(nn.Module):

    def __init__(self, ch: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class CNNResidual(nn.Module):

    def __init__(self, in_ch: int = 12, base_ch: int = 64,
                 n_blocks: int = 4, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.out_channels = base_ch

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=15, stride=2,
                      padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        block_cls = ResidualBlock1D if use_residual else PlainBlock1D
        self.blocks = nn.Sequential(
            *[block_cls(base_ch) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


if __name__ == "__main__":
    for res in (True, False):
        m = CNNResidual(use_residual=res)
        x = torch.randn(4, 12, 1000)
        y = m(x)
        print(f"residual={res}: CNN output: {tuple(y.shape)}")
        assert y.shape[0] == 4 and y.shape[1] == m.out_channels
        assert y.shape[2] == 250, f"Expected 250, got {y.shape[2]}"
    print("OK — CNN+Residual shape contract holds.")
