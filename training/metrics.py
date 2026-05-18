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
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score

    # Convert to numpy arrays on CPU
    y_true_np = y_true.detach().cpu().numpy()
    y_logits_np = y_logits.detach().cpu().numpy()
    
    # Apply sigmoid to logits to get probabilities
    y_probs = 1.0 / (1.0 + np.exp(-y_logits_np))
    y_pred = (y_probs >= 0.5).astype(np.float32)

    n_classes = y_true_np.shape[1]
    
    # Safe macro ROC-AUC calculation
    auc_scores = []
    for i in range(n_classes):
        # We need both classes (0 and 1) in y_true to calculate ROC-AUC
        if len(np.unique(y_true_np[:, i])) == 2:
            try:
                score = roc_auc_score(y_true_np[:, i], y_probs[:, i])
                auc_scores.append(score)
            except ValueError:
                pass
                
    if len(auc_scores) > 0:
        macro_auc = float(np.mean(auc_scores))
    else:
        macro_auc = 0.5  # Baseline AUC for uncalculated classes

    # Safe macro F1 calculation
    f1_scores = []
    for i in range(n_classes):
        try:
            score = f1_score(y_true_np[:, i], y_pred[:, i], zero_division=0)
            f1_scores.append(score)
        except Exception:
            f1_scores.append(0.0)
            
    macro_f1 = float(np.mean(f1_scores))

    return {"macro_auc": macro_auc, "macro_f1": macro_f1}


if __name__ == "__main__":
    y_t = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0]])
    y_l = torch.tensor([[2.0, -2.0, -3.0, -4.0, -5.0],
                        [-2.0, 2.0, -3.0, -4.0, -5.0]])
    res = compute_metrics(y_t, y_l)
    print("Smoke test results:", res)

