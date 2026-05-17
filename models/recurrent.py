"""Block 4 — Recurrent (BiLSTM / BiGRU) (Person B).

COURSE JUSTIFICATION (W7):
    ECG diagnosis depends not only on local morphology but also on
    TEMPORAL patterns (arrhythmia, rhythm irregularity). The LSTM/GRU
    covered in W7 solve the vanishing-gradient problem via gating and
    capture long-term dependencies (Hochreiter & Schmidhuber 1997;
    Cho 2014). Since diagnosis is done OFFLINE (not real-time), using a
    bidirectional model is justified — W7: a BiRNN sees the whole
    sequence.

    Ablation axes: rnn_type ∈ {lstm, gru}, bidirectional ∈ {True, False}.

Contract (CONTRACT.md):
    Input : CNN output [B, C, L'] → transposed internally to [B, L', C]
    Output: [B, L', H]  (H = 2*hidden if bidirectional)
"""

from __future__ import annotations
import torch
import torch.nn as nn


class RecurrentBlock(nn.Module):
    """BiLSTM or BiGRU wrapper.

    Args:
        in_features: CNN output channel count (C).
        hidden: hidden size.
        rnn_type: "lstm" | "gru".
        bidirectional: bidirectional or not.
        num_layers: stacked RNN layers.
    """

    def __init__(self, in_features: int, hidden: int = 128,
                 rnn_type: str = "lstm", bidirectional: bool = True,
                 num_layers: int = 1):
        super().__init__()
        rnn_type = rnn_type.lower()
        assert rnn_type in {"lstm", "gru"}
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=in_features, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_features = hidden * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L'] -> [B, L', out_features]."""
        x = x.transpose(1, 2)          # [B, L', C]
        out, _ = self.rnn(x)           # [B, L', out_features]
        return out


if __name__ == "__main__":
    for rt in ("lstm", "gru"):
        for bi in (True, False):
            m = RecurrentBlock(in_features=64, hidden=128,
                               rnn_type=rt, bidirectional=bi)
            x = torch.randn(4, 64, 50)        # [B, C, L']
            y = m(x)
            exp = 128 * (2 if bi else 1)
            print(f"{rt} bidir={bi}: {tuple(y.shape)} (H={exp})")
            assert y.shape == (4, 50, exp)
    print("OK — Recurrent shape contract holds.")
