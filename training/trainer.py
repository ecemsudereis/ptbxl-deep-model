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
    opt_name = cfg.optimizer.lower()
    # Filter weight parameters vs bias parameters if needed, but applying weight decay
    # to all parameters is standard unless specified otherwise. We apply it to all parameters.
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def l1_penalty(model: nn.Module) -> torch.Tensor:
    """W4: L1 norm of the weights (excluding bias)."""
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_loss += torch.sum(torch.abs(param))
    return l1_loss


def train(model: nn.Module, train_loader, val_loader,
          cfg: TrainConfig) -> dict:
    """Full training: (opt.) DAE pretraining → main training + early stop.

    Returns:
        {"best_val_metrics": dict, "best_epoch": int, "history": list}
    """
    import copy
    from training.metrics import compute_metrics

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    model = model.to(device)

    # 1. Optional DAE pretraining phase (unsupervised reconstruction)
    if cfg.dae_pretrain_epochs > 0 and getattr(model, "dae", None) is not None:
        print(f"\n--- Starting DAE Pretraining ({cfg.dae_pretrain_epochs} epochs) ---")
        dae_params = list(model.dae.parameters())
        if len(dae_params) == 0:
            print("Warning: DAE has no learnable parameters (placeholder). Skipping DAE pretraining steps.")
        else:
            dae_optimizer = torch.optim.Adam(dae_params, lr=cfg.lr)
            model.train()
            for epoch in range(1, cfg.dae_pretrain_epochs + 1):
                total_recon_loss = 0.0
                num_batches = 0
                for x, _ in train_loader:
                    x = x.to(device)
                    dae_optimizer.zero_grad()
                    loss = model.dae.reconstruction_loss(x)
                    loss.backward()
                    dae_optimizer.step()
                    total_recon_loss += loss.item()
                    num_batches += 1
                avg_loss = total_recon_loss / max(num_batches, 1)
                print(f"DAE Pretrain Epoch {epoch:02d}/{cfg.dae_pretrain_epochs:02d} | Reconstruction Loss: {avg_loss:.6f}")
            print("--- DAE Pretraining Complete ---\n")

    # 2. Main training phase
    print(f"--- Starting Main Training ({cfg.epochs} epochs) ---")
    optimizer = build_optimizer(model, cfg)

    # LR schedule (W5)
    scheduler = None
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_schedule == "linear":
        lr_lambda = lambda epoch: 1.0 - (epoch / cfg.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_val_metrics = {}
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        # Training epoch
        model.train()
        train_loss = 0.0
        train_batches = 0
        all_train_logits = []
        all_train_labels = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, aux = model(x)

            # W4: Label smoothing
            if cfg.label_smoothing > 0.0:
                y_smoothed = y * (1.0 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
                bce_loss = criterion(logits, y_smoothed)
            else:
                bce_loss = criterion(logits, y)

            # W4: L1 regularization
            l1_loss = l1_penalty(model) * cfg.l1_lambda

            # W8: VAE KL regularization
            kl_loss = torch.tensor(0.0, device=device)
            if getattr(model, "vae", None) is not None and "vae" in aux:
                mu, logvar = aux["vae"]
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * cfg.vae_beta

            loss = bce_loss + l1_loss + kl_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            all_train_logits.append(logits.detach())
            all_train_labels.append(y.detach())

        avg_train_loss = train_loss / max(train_batches, 1)
        all_train_logits = torch.cat(all_train_logits, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        train_metrics = compute_metrics(all_train_labels, all_train_logits)

        # Validation epoch
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_val_logits = []
        all_val_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, aux = model(x)

                bce_loss = criterion(logits, y)
                l1_loss = l1_penalty(model) * cfg.l1_lambda

                kl_loss = torch.tensor(0.0, device=device)
                if getattr(model, "vae", None) is not None and "vae" in aux:
                    mu, logvar = aux["vae"]
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * cfg.vae_beta

                loss = bce_loss + l1_loss + kl_loss
                val_loss += loss.item()
                val_batches += 1
                all_val_logits.append(logits)
                all_val_labels.append(y)

        avg_val_loss = val_loss / max(val_batches, 1)
        all_val_logits = torch.cat(all_val_logits, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)
        val_metrics = compute_metrics(all_val_labels, all_val_logits)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_log = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_auc": train_metrics["macro_auc"],
            "train_f1": train_metrics["macro_f1"],
            "val_loss": avg_val_loss,
            "val_auc": val_metrics["macro_auc"],
            "val_f1": val_metrics["macro_f1"],
            "lr": current_lr
        }
        history.append(epoch_log)

        print(f"Epoch {epoch:02d}/{cfg.epochs:02d} | Train Loss: {avg_train_loss:.4f} (AUC: {train_metrics['macro_auc']:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_metrics['macro_auc']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | "
              f"LR: {current_lr:.6f}")

        # W5: Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        # W4: Early stopping (Algorithm 7.1)
        # We monitor validation macro-AUC (maximizing it)
        current_val_auc = val_metrics["macro_auc"]
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}! No improvement in validation AUC for {cfg.early_stop_patience} epochs.")
                break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model state from epoch {best_epoch} with Val AUC: {best_val_auc:.4f}")

    return {
        "best_val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
        "history": history
    }


def load_config(yaml_path: str) -> TrainConfig:
    """Load baseline.yaml and parse it into a TrainConfig object."""
    import yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    t_data = data.get("train", {})
    r_data = data.get("regularization", {})
    a_data = data.get("aux", {})
    m_data = data.get("model", {})

    return TrainConfig(
        epochs=t_data.get("epochs", 50),
        lr=t_data.get("lr", 1e-3),
        batch_size=t_data.get("batch_size", 64),
        optimizer=t_data.get("optimizer", "adam"),
        weight_decay=r_data.get("weight_decay", 1e-4),
        l1_lambda=r_data.get("l1_lambda", 0.0),
        label_smoothing=r_data.get("label_smoothing", 0.0),
        vae_beta=a_data.get("vae_beta", 1.0),
        lr_schedule=t_data.get("lr_schedule", "cosine"),
        early_stop_patience=r_data.get("early_stop_patience", 8),
        dae_pretrain_epochs=a_data.get("dae_pretrain_epochs", 0),
        device=t_data.get("device", "cuda"),
        flags=m_data
    )


# --- Dummy end-to-end smoke test ---------------------------------------
# Wires up end-to-end with loaded config and mock/dummy loaders
if __name__ == "__main__":
    import sys
    import os
    # Add root of the project to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.full_model import FullModel
    from data.dataset import get_loaders

    print("Loading baseline config...")
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "baseline.yaml")
    
    cfg = load_config(config_path)
    
    # Overrides for rapid smoke test verification
    cfg.epochs = 5
    cfg.dae_pretrain_epochs = 2
    cfg.early_stop_patience = 3
    cfg.l1_lambda = 1e-5  # turn on L1 penalty to verify L1 calculation
    
    print("Setting up model and dummy loaders...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set to run on standard device
    cfg.device = device
    
    # Initialize full model with config flags
    model = FullModel(
        use_dae=cfg.flags.get("use_dae", True),
        use_residual=cfg.flags.get("use_residual", True),
        rnn_type=cfg.flags.get("rnn_type", "lstm"),
        bidirectional=cfg.flags.get("bidirectional", True),
        use_attention=cfg.flags.get("use_attention", True),
        use_vae=cfg.flags.get("use_vae", True),
        cnn_base_ch=cfg.flags.get("cnn_base_ch", 64),
        rnn_hidden=cfg.flags.get("rnn_hidden", 128)
    ).to(device)

    # Use dummy loader
    tr, va, te = get_loaders(batch_size=cfg.batch_size, dummy=True, num_workers=0)
    
    print("\n--- Running End-to-End Dummy Train Loop ---")
    results = train(model, tr, va, cfg)
    
    print("\n--- Training Results ---")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Best Val Metrics: {results['best_val_metrics']}")
    
    # Verify that the loss actually decreased
    first_loss = results["history"][0]["train_loss"]
    last_loss = results["history"][-1]["train_loss"]
    print(f"\nLoss progression: First Epoch Loss = {first_loss:.4f} -> Last Epoch Loss = {last_loss:.4f}")
    
    if last_loss < first_loss:
        print("SUCCESS: Loss successfully decreased over training epochs!")
    else:
        print("WARNING: Loss did not decrease. Check model and optimization flow.")

