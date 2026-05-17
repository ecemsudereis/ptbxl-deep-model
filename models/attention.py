"""Block 5 — Temporal Attention (Person B).

COURSE JUSTIFICATION (W7/W8):
    Not every time step matters equally for diagnosis; an arrhythmia may
    appear in only a few beats. Attention — introduced in W7 as the fix
    for the seq2seq bottleneck and at the end of W8 as "content-based
    memory access" — pools the recurrent outputs with learned weights
    (Bahdanau et al. 2015). Bonus: the weights are INTERPRETABLE — they
    show which segment drove the diagnosis (visualized in the paper).

    Ablation: if use_attention=False, falls back to simple mean-pooling.

Contract (CONTRACT.md):
    forward(x: [B, L', H]) -> (pooled: [B, H], weights: [B, L'])
"""

from __future__ import annotations
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention pooling over the time axis.

    Args:
        hidden: input feature size (recurrent out_features).
        attn_dim: intermediate attention projection size.
    """

    def __init__(self, hidden: int, attn_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor):
        """x: [B, L', H] -> (pooled [B,H], weights [B,L'])."""
        e = torch.tanh(self.proj(x))           # [B, L', attn_dim]
        scores = self.score(e).squeeze(-1)     # [B, L']
        weights = torch.softmax(scores, dim=1) # [B, L']
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, H]
        return pooled, weights


class MeanPool(nn.Module):
    """Ablation baseline: plain mean instead of attention."""

    def forward(self, x: torch.Tensor):
        return x.mean(dim=1), None


if __name__ == "__main__":
    m = TemporalAttention(hidden=256)
    x = torch.randn(4, 50, 256)
    p, w = m(x)
    print("pooled:", tuple(p.shape), "| weights:", tuple(w.shape))
    assert p.shape == (4, 256) and w.shape == (4, 50)
    assert torch.allclose(w.sum(1), torch.ones(4), atol=1e-5)
    print("OK — Attention shape + softmax contract holds.")
