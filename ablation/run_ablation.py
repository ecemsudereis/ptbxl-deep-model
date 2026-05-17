"""Ablation Runner (Person D) — the +15 of the bonus.

SCIENTIFIC PRINCIPLE: All variants are compared on the SAME folds
(CONTRACT §3) with the SAME metrics (CONTRACT §4). We remove one block
at a time and measure the macro-AUC/F1 drop → proving each block's
contribution.

ABLATION MATRIX (will be Table 2 in the paper):

    #  Variant                         Changed flag
    0  Full model (6 blocks)           — (reference)
    1  − Denoising AE                  use_dae=False
    2  − Residual                      use_residual=False
    3  − Attention (mean-pool)         use_attention=False
    4  − VAE                           use_vae=False
    5  LSTM → GRU                      rnn_type=gru
    6  Bidirectional → unidirectional  bidirectional=False
    7  CNN only (no RNN/AE/Attn)       minimal baseline

Optional extra: pairwise removals (e.g. −DAE −VAE) — if time permits.
"""

from __future__ import annotations

# Ablation matrix: overrides applied on top of the baseline config.
ABLATION_MATRIX = [
    {"name": "full_6block",      "override": {}},
    {"name": "no_dae",           "override": {"use_dae": False}},
    {"name": "no_residual",      "override": {"use_residual": False}},
    {"name": "no_attention",     "override": {"use_attention": False}},
    {"name": "no_vae",           "override": {"use_vae": False}},
    {"name": "gru_instead_lstm", "override": {"rnn_type": "gru"}},
    {"name": "unidirectional",   "override": {"bidirectional": False}},
    {"name": "cnn_only",         "override": {"use_dae": False,
                                              "use_attention": False,
                                              "use_vae": False}},
]


def run_one(name: str, override: dict) -> dict:
    """Train one variant + evaluate on the test fold.

    Returns:
        {"name": str, "macro_auc": float, "macro_f1": float, ...}
    """
    # TODO(D): load baseline.yaml → apply override → trainer.train →
    #          compute_metrics on the test fold (10) → store result.
    raise NotImplementedError("Person D: single variant run.")


def run_all(out_csv: str = "ablation/results/ablation_summary.csv") -> None:
    """Run the whole matrix, save the result table as CSV + plot."""
    # TODO(D): loop + results to a pandas DataFrame → CSV; matplotlib bar
    #          chart (AUC drop when each block is removed).
    raise NotImplementedError("Person D: full matrix + table/plot.")


if __name__ == "__main__":
    print("Ablation matrix —", len(ABLATION_MATRIX), "variants:")
    for r in ABLATION_MATRIX:
        print(f"  {r['name']:<20} override={r['override']}")
