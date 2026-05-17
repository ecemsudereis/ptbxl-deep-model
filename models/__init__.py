"""Architecture layer (Person B).

6 distinct blocks, each an independent nn.Module:
    dae.py          — Denoising Autoencoder      (W8)
    cnn_residual.py — 1D CNN + Residual          (W6 + W5)
    recurrent.py    — BiLSTM / BiGRU             (W7)
    attention.py    — Temporal Attention         (W7/W8)
    vae.py          — Variational Autoencoder    (W8)
    full_model.py   — main model fusing all      (ablation flags)

Each block conforms to the shapes in CONTRACT.md and has its own
smoke test.
"""
