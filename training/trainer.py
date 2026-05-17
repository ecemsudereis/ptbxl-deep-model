"""Training loop (Person C).

COURSE LINKS (directly feeds the paper's "hyperparameter tuning &
regularization" section):

    Optimizer (W5):   Adam / RMSProp / SGD+momentum — chosen from config
    LR schedule (W5): linear decay / cosine — from config
    Regularization arsenal (W4):
        - L2 weight decay      (optimizer weight_decay)
        - L1 penalty           (term added to the loss)
        - Dropout              (in the model, p from config)
        - BatchNorm            (in the model)
        - Early stopping       (patience — W4 Algorithm 7.1)
        - Label smoothing      (softened BCE targets)
    Hyperparameter tuning (W2): on the validation fold (9); the test
        fold (10) NEVER enters tuning.

Total loss:
    L = BCE(logits, y)
        + l1_lambda * Σ|w|              (W4 L1)
        + vae_beta  * KL(mu, logvar)    (W8, if use_vae)
    + DAE pretraining as a separate phase (optional): denoising MSE.

This file is a SKELETON — logic is TODO(C). But it runs end-to-end with
dummies.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 64
    optimizer: str = "adam"          # adam | rmsprop | sgd  (W5)
    weight_decay: float = 1e-4       # L2 (W4)
    l1_lambda: float = 0.0           # L1 (W4)
    label_smoothing: float = 0.0     # (W4)
    vae_beta: float = 1.0            # KL weight (W8)
    lr_schedule: str = "cosine"      # none | linear | cosine (W5)
    early_stop_patience: int = 8     # (W4 Algorithm 7.1)
    dae_pretrain_epochs: int = 0     # >0 → DAE pretraining first
    device: str = "cuda"
    flags: dict = field(default_factory=dict)  # FullModel ablation flags


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    """W5: build optimizer from config (Adam/RMSProp/SGD+momentum)."""
    # TODO(C): return torch.optim.* per cfg.optimizer, apply weight_decay.
    raise NotImplementedError("Person C: optimizer factory.")


def l1_penalty(model: nn.Module) -> torch.Tensor:
    """W4: L1 norm of the weights (excluding bias)."""
    # TODO(C): Σ|w| — weight parameters only.
    raise NotImplementedError("Person C: L1 penalty.")


def train(model: nn.Module, train_loader, val_loader,
          cfg: TrainConfig) -> dict:
    """Full training: (opt.) DAE pretraining → main training + early stop.

    Returns:
        {"best_val_metrics": dict, "best_epoch": int, "history": list}
    """
    # TODO(C): epoch loop, total loss (BCE + L1 + VAE-KL), per-epoch val
    #          metric, early stopping, best checkpoint.
    raise NotImplementedError("Person C: main training loop.")


# --- Dummy end-to-end smoke test ---------------------------------------
# Without real trainer logic, verifies that the model + data interfaces
# wire up end-to-end. Anyone can run this.
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models.full_model import FullModel
    from data.dataset import get_loaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FullModel().to(device)
    tr, va, te = get_loaders(batch_size=8, dummy=True, num_workers=0)

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    x, y = next(iter(tr))
    x, y = x.to(device), y.to(device)
    logits, aux = model(x)
    loss = crit(logits, y)
    if aux.get("vae") is not None:
        mu, logvar = aux["vae"]
        loss = loss + (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
    opt.zero_grad(); loss.backward(); opt.step()

    print(f"Dummy end-to-end OK | device={device} | loss={loss.item():.4f}")
    print("Model <-> Data <-> Loss chain wired. Person C fills the logic.")
