from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.full_model import FullModel
from training.metrics import compute_metrics
from training.trainer import TrainConfig, build_optimizer, l1_penalty, load_config


torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


ABLATION_MATRIX = [
    {"name": "full_6block", "override": {}},
    {"name": "no_dae", "override": {"use_dae": False}},
    {"name": "no_residual", "override": {"use_residual": False}},
    {"name": "no_attention", "override": {"use_attention": False}},
    {"name": "no_vae", "override": {"use_vae": False}},
    {"name": "gru_instead_lstm", "override": {"rnn_type": "gru"}},
    {"name": "unidirectional", "override": {"bidirectional": False}},
    {"name": "minimal_supported", "override": {"use_dae": False, "use_residual": False, "use_attention": False, "use_vae": False}},
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class LimitedLoader:
    def __init__(self, loader, max_batches: int | None):
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = loader.dataset

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            yield batch

    def __len__(self):
        if self.max_batches is None:
            return len(self.loader)
        return min(len(self.loader), self.max_batches)


def make_dummy_loaders(batch_size: int):
    from torch.utils.data import DataLoader, TensorDataset
    def loader(n: int, shuffle: bool):
        x = torch.randn(n, 12, 1000, dtype=torch.float32)
        y = (torch.rand(n, 5) > 0.7).float()
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)
    return loader(64, True), loader(32, False), loader(32, False)


def make_loaders(batch_size: int, dummy: bool, num_workers: int, max_batches: int | None):
    if dummy:
        train_loader, val_loader, test_loader = make_dummy_loaders(batch_size)
    else:
        try:
            from data.dataset import get_loaders
            train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size, dummy=False, num_workers=num_workers)
        except FileNotFoundError as exc:
            raise FileNotFoundError("PTB-XL data is not available. Put archive.zip under data/raw/ and run `python -m data.download`, or use `--dummy` only for a smoke test.") from exc
    return LimitedLoader(train_loader, max_batches), val_loader, test_loader


def cfg_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return {
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "optimizer": cfg.optimizer,
        "weight_decay": cfg.weight_decay,
        "l1_lambda": cfg.l1_lambda,
        "label_smoothing": cfg.label_smoothing,
        "vae_beta": cfg.vae_beta,
        "lr_schedule": cfg.lr_schedule,
        "early_stop_patience": cfg.early_stop_patience,
        "dae_pretrain_epochs": cfg.dae_pretrain_epochs,
        "device": cfg.device,
        "flags": dict(cfg.flags),
    }


def apply_override(cfg: TrainConfig, override: dict[str, Any]) -> TrainConfig:
    updated = copy.deepcopy(cfg)
    updated.flags = dict(updated.flags)
    for key, value in override.items():
        if hasattr(updated, key):
            setattr(updated, key, value)
        else:
            updated.flags[key] = value
    return updated


def build_model(cfg: TrainConfig) -> FullModel:
    flags = dict(cfg.flags)
    return FullModel(
        use_dae=flags.get("use_dae", True),
        use_residual=flags.get("use_residual", True),
        rnn_type=flags.get("rnn_type", "lstm"),
        bidirectional=flags.get("bidirectional", True),
        use_attention=flags.get("use_attention", True),
        use_vae=flags.get("use_vae", True),
        cnn_base_ch=int(flags.get("cnn_base_ch", 64)),
        rnn_hidden=int(flags.get("rnn_hidden", 128)),
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, amp: bool) -> dict[str, float]:
    model.eval()
    logits_list = []
    labels_list = []
    total_loss = 0.0
    total_batches = 0
    criterion = nn.BCEWithLogitsLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            logits, _ = model(x)
            loss = criterion(logits, y)
        total_loss += float(loss.item())
        total_batches += 1
        logits_list.append(logits.detach().cpu())
        labels_list.append(y.detach().cpu())
    if not logits_list:
        return {"macro_auc": 0.5, "macro_f1": 0.0, "loss": 0.0}
    metrics = compute_metrics(torch.cat(labels_list, dim=0), torch.cat(logits_list, dim=0))
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics


@torch.no_grad()
def save_attention_plot(model: torch.nn.Module, loader, device: torch.device, out_path: Path, amp: bool) -> None:
    model.eval()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            _, aux = model(x)
        weights = aux.get("attn_weights") if isinstance(aux, dict) else None
        if weights is None:
            return
        values = weights[0].detach().float().cpu().numpy()
        if values.ndim > 1:
            values = values.squeeze()
        import matplotlib.pyplot as plt
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 3))
        plt.plot(values)
        plt.xlabel("Temporal feature index")
        plt.ylabel("Attention weight")
        plt.title("Example temporal attention distribution")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return


def vae_kl(aux: dict[str, Any], device: torch.device, beta: float) -> torch.Tensor:
    if "vae" not in aux:
        return torch.tensor(0.0, device=device)
    mu, logvar = aux["vae"]
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * beta


def pretrain_dae(model: torch.nn.Module, loader, cfg: TrainConfig, device: torch.device, amp: bool) -> None:
    if cfg.dae_pretrain_epochs <= 0 or getattr(model, "dae", None) is None:
        return
    params = list(model.dae.parameters())
    if not params:
        return
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    print(f"--- Starting DAE Pretraining ({cfg.dae_pretrain_epochs} epochs) ---")
    for epoch in range(1, cfg.dae_pretrain_epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                loss = model.dae.reconstruction_loss(x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            batches += 1
        print(f"DAE Pretrain Epoch {epoch:02d}/{cfg.dae_pretrain_epochs:02d} | Reconstruction Loss: {total / max(batches, 1):.6f}")
    print("--- DAE Pretraining Complete ---")


def train_model(model: torch.nn.Module, train_loader, val_loader, cfg: TrainConfig, device: torch.device, amp: bool) -> dict[str, Any]:
    pretrain_dae(model, train_loader, cfg, device, amp)
    print(f"--- Starting Main Training ({cfg.epochs} epochs) ---")
    optimizer = build_optimizer(model, cfg)
    scheduler = None
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_schedule == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(0.0, 1.0 - epoch / max(cfg.epochs, 1)))
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best_state = None
    best_val_metrics = {}
    best_val_auc = -1.0
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        train_logits = []
        train_labels = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits, aux = model(x)
                targets = y * (1.0 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing if cfg.label_smoothing > 0.0 else y
                loss = criterion(logits, targets) + l1_penalty(model) * cfg.l1_lambda + vae_kl(aux, device, cfg.vae_beta)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
            train_batches += 1
            train_logits.append(logits.detach().cpu())
            train_labels.append(y.detach().cpu())
        train_metrics = compute_metrics(torch.cat(train_labels, dim=0), torch.cat(train_logits, dim=0)) if train_logits else {"macro_auc": 0.5, "macro_f1": 0.0}
        val_metrics = evaluate(model, val_loader, device, amp)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_batches, 1),
            "train_auc": train_metrics["macro_auc"],
            "train_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["macro_auc"],
            "val_f1": val_metrics["macro_f1"],
            "lr": current_lr,
        }
        history.append(row)
        print(f"Epoch {epoch:02d}/{cfg.epochs:02d} | Train Loss: {row['train_loss']:.4f} (AUC: {row['train_auc']:.4f}) | Val Loss: {row['val_loss']:.4f} | Val AUC: {row['val_auc']:.4f} | Val F1: {row['val_f1']:.4f} | LR: {current_lr:.6f}")
        if scheduler is not None:
            scheduler.step()
        if val_metrics["macro_auc"] > best_val_auc:
            best_val_auc = val_metrics["macro_auc"]
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Restored best model state from epoch {best_epoch} with Val AUC: {best_val_auc:.4f}")
    return {"best_val_metrics": best_val_metrics, "best_epoch": best_epoch, "history": history}


def save_history(history: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_results_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "macro_auc", "macro_f1", "auc_drop", "f1_drop", "val_macro_auc", "val_macro_f1", "test_loss", "best_epoch", "train_time_sec", "params", "override"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_drop_plot(rows: list[dict[str, Any]], metric_key: str, out_path: Path, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt
    names = [row["name"] for row in rows if row["name"] != "full_6block"]
    values = [float(row[metric_key]) for row in rows if row["name"] != "full_6block"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.bar(names, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one(name: str, override: dict[str, Any], base_cfg: TrainConfig, out_dir: str | Path, dummy: bool, seed: int, num_workers: int, save_attention: bool, amp: bool, max_batches: int | None) -> dict[str, Any]:
    print(f"Running ablation variant: {name}")
    set_seed(seed)
    cfg = apply_override(base_cfg, override)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    if dummy:
        cfg.flags = dict(cfg.flags)
        cfg.flags["cnn_base_ch"] = min(int(cfg.flags.get("cnn_base_ch", 64)), 8)
        cfg.flags["rnn_hidden"] = min(int(cfg.flags.get("rnn_hidden", 128)), 16)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_loader, val_loader, test_loader = make_loaders(cfg.batch_size, dummy, num_workers, max_batches)
    model = build_model(cfg).to(device)
    params = count_params(model)
    start = time.time()
    result = train_model(model, train_loader, val_loader, cfg, device, amp)
    train_time = time.time() - start
    test_metrics = evaluate(model, test_loader, device, amp)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_history(result.get("history", []), out / f"history_{name}.csv")
    with (out / f"config_{name}.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_to_dict(cfg), f, indent=2)
    if save_attention:
        save_attention_plot(model, test_loader, device, out / "attention_example.png", amp)
    best_val = result.get("best_val_metrics", {}) or {}
    row = {
        "name": name,
        "macro_auc": float(test_metrics.get("macro_auc", 0.5)),
        "macro_f1": float(test_metrics.get("macro_f1", 0.0)),
        "auc_drop": 0.0,
        "f1_drop": 0.0,
        "val_macro_auc": float(best_val.get("macro_auc", 0.5)),
        "val_macro_f1": float(best_val.get("macro_f1", 0.0)),
        "test_loss": float(test_metrics.get("loss", 0.0)),
        "best_epoch": int(result.get("best_epoch", 0) or 0),
        "train_time_sec": round(train_time, 2),
        "params": int(params),
        "override": json.dumps(override, sort_keys=True),
    }
    print(f"Finished {name}: test macro-AUC={row['macro_auc']:.4f}, test macro-F1={row['macro_f1']:.4f}")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def select_variants(only: str | None, variants: str | None) -> list[dict[str, Any]]:
    matrix = ABLATION_MATRIX
    if only:
        names = [only]
    elif variants:
        names = [x.strip() for x in variants.split(",") if x.strip()]
    else:
        return matrix
    lookup = {item["name"]: item for item in matrix}
    missing = [name for name in names if name not in lookup]
    if missing:
        valid = ", ".join(lookup)
        raise ValueError(f"Unknown ablation variant(s): {', '.join(missing)}. Valid variants: {valid}")
    return [lookup[name] for name in names]


def run_all(out_csv: str = "ablation/results/ablation_summary.csv", config_path: str = "configs/baseline.yaml", dummy: bool = False, seed: int = 42, num_workers: int = 0, only: str | None = None, variants: str | None = None, epochs: int | None = None, batch_size: int | None = None, device: str | None = None, dae_pretrain_epochs: int | None = None, cnn_base_ch: int | None = None, rnn_hidden: int | None = None, lr: float | None = None, patience: int | None = None, amp: bool = True, max_batches: int | None = None) -> list[dict[str, Any]]:
    base_cfg = load_config(config_path)
    if epochs is not None:
        base_cfg.epochs = epochs
    if batch_size is not None:
        base_cfg.batch_size = batch_size
    if device is not None:
        base_cfg.device = device
    if dae_pretrain_epochs is not None:
        base_cfg.dae_pretrain_epochs = dae_pretrain_epochs
    if lr is not None:
        base_cfg.lr = lr
    if patience is not None:
        base_cfg.early_stop_patience = patience
    base_cfg.flags = dict(base_cfg.flags)
    if cnn_base_ch is not None:
        base_cfg.flags["cnn_base_ch"] = cnn_base_ch
    if rnn_hidden is not None:
        base_cfg.flags["rnn_hidden"] = rnn_hidden
    selected = select_variants(only, variants)
    out_path = Path(out_csv)
    out_dir = out_path.parent
    rows = []
    print("Selected ablation variants:")
    for item in selected:
        print(f"  {item['name']:<20} override={item['override']}")
    for item in selected:
        rows.append(run_one(item["name"], item["override"], base_cfg=copy.deepcopy(base_cfg), out_dir=out_dir, dummy=dummy, seed=seed, num_workers=num_workers, save_attention=item["name"] == "full_6block" or only is not None, amp=amp, max_batches=max_batches))
        write_results_csv(rows, out_path)
    baseline = next((row for row in rows if row["name"] == "full_6block"), rows[0] if rows else None)
    baseline_auc = baseline["macro_auc"] if baseline else 0.0
    baseline_f1 = baseline["macro_f1"] if baseline else 0.0
    for row in rows:
        row["auc_drop"] = float(baseline_auc - row["macro_auc"])
        row["f1_drop"] = float(baseline_f1 - row["macro_f1"])
    write_results_csv(rows, out_path)
    if len(rows) > 1 and baseline is not None:
        save_drop_plot(rows, "auc_drop", out_dir / "auc_drop.png", "Macro-AUC drop relative to the full model", "Macro-AUC drop")
        save_drop_plot(rows, "f1_drop", out_dir / "f1_drop.png", "Macro-F1 drop relative to the full model", "Macro-F1 drop")
    print(f"Saved ablation summary to {out_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--out-csv", default="ablation/results/ablation_summary.csv")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--only", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dae-pretrain-epochs", type=int, default=None)
    parser.add_argument("--cnn-base-ch", type=int, default=None)
    parser.add_argument("--rnn-hidden", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all(
        out_csv=args.out_csv,
        config_path=args.config,
        dummy=args.dummy,
        seed=args.seed,
        num_workers=args.num_workers,
        only=args.only,
        variants=args.variants,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        dae_pretrain_epochs=args.dae_pretrain_epochs,
        cnn_base_ch=args.cnn_base_ch,
        rnn_hidden=args.rnn_hidden,
        lr=args.lr,
        patience=args.patience,
        amp=not args.no_amp,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
