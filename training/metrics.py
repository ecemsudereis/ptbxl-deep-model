"""Metrics (Person C) — CONTRACT §4.

Single signature, used throughout the project:
    compute_metrics(y_true [B,5], y_logits [B,5]) -> dict
The ablation compares all variants with these metrics.
"""

from __future__ import annotations
import torch


def compute_metrics(y_true: torch.Tensor, y_logits: torch.Tensor) -> dict:
    """Multi-label macro-AUC and macro-F1.

    Args:
        y_true:   [N, 5] 0/1 ground-truth labels.
        y_logits: [N, 5] raw logits (no sigmoid applied).

    Returns:
        {"macro_auc": float, "macro_f1": float}
    """
    # TODO(C): sklearn.metrics.roc_auc_score(average="macro") +
    #          f1_score(average="macro", threshold=0.5).
    #          Handle the case where a class is single-valued (AUC may
    #          be NaN) safely.
    raise NotImplementedError("Person C: macro-AUC + macro-F1.")


if __name__ == "__main__":
    print("metrics.py — skeleton. Signature: compute_metrics(y_true, y_logits)")
