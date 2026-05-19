from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

from models.full_model import FullModel
from training.metrics import compute_metrics
from training.trainer import TrainConfig, load_config, train


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


def make_dummy_loaders(batch_size: int):
    from torch.utils.data import DataLoader, TensorDataset
    def loader(n: int, shuffle: bool):
        x = torch.randn(n, 12, 1000, dtype=torch.float32)
        y = (torch.rand(n, 5) > 0.7).float()
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)
    return loader(16, True), loader(8, False), loader(8, False)


def make_loaders(batch_size: int, dummy: bool, num_workers: int):
    if dummy:
        return make_dummy_loaders(batch_size)
    from data.dataset import get_loaders
    return get_loaders(batch_size=batch_size, dummy=False, num_workers=num_workers)


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
        cnn_base_ch=flags.get("cnn_base_ch", 64),
        rnn_hidden=flags.get("rnn_hidden", 128),
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    logits_list = []
    labels_list = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        logits_list.append(logits.detach().cpu())
        labels_list.append(y.detach().cpu())
    if not logits_list:
        return {"macro_auc": 0.5, "macro_f1": 0.0}
    return compute_metrics(torch.cat(labels_list, dim=0), torch.cat(logits_list, dim=0))


@torch.no_grad()
def save_attention_plot(model: torch.nn.Module, loader, device: torch.device, out_path: Path) -> None:
    model.eval()
    for x, _ in loader:
        x = x.to(device)
        _, aux = model(x)
        weights = aux.get("attn_weights") if isinstance(aux, dict) else None
        if weights is None:
            return
        values = weights[0].detach().cpu().numpy()
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
    fieldnames = ["name", "macro_auc", "macro_f1", "auc_drop", "f1_drop", "best_epoch", "override"]
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


def run_one(
    name: str,
    override: dict[str, Any],
    base_cfg: TrainConfig | None = None,
    config_path: str = "configs/baseline.yaml",
    out_dir: str | Path = "ablation/results",
    dummy: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    save_attention: bool = True,
) -> dict[str, Any]:
    print(f"Running ablation variant: {name}")
    set_seed(seed)
    cfg = copy.deepcopy(base_cfg) if base_cfg is not None else load_config(config_path)
    cfg = apply_override(cfg, override)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    if dummy:
        cfg.flags = dict(cfg.flags)
        cfg.flags["cnn_base_ch"] = min(int(cfg.flags.get("cnn_base_ch", 64)), 8)
        cfg.flags["rnn_hidden"] = min(int(cfg.flags.get("rnn_hidden", 128)), 16)
    train_loader, val_loader, test_loader = make_loaders(cfg.batch_size, dummy, num_workers)
    model = build_model(cfg).to(device)
    result = train(model, train_loader, val_loader, cfg)
    test_metrics = evaluate(model, test_loader, device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_history(result.get("history", []), out / f"history_{name}.csv")
    with (out / f"config_{name}.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_to_dict(cfg), f, indent=2)
    if save_attention:
        save_attention_plot(model, test_loader, device, out / "attention_example.png")
    row = {
        "name": name,
        "macro_auc": float(test_metrics.get("macro_auc", 0.5)),
        "macro_f1": float(test_metrics.get("macro_f1", 0.0)),
        "best_epoch": int(result.get("best_epoch", 0) or 0),
        "override": json.dumps(override, sort_keys=True),
    }
    print(f"Finished {name}: test macro-AUC={row['macro_auc']:.4f}, test macro-F1={row['macro_f1']:.4f}")
    return row


def run_all(
    out_csv: str = "ablation/results/ablation_summary.csv",
    config_path: str = "configs/baseline.yaml",
    dummy: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    only: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    dae_pretrain_epochs: int | None = None,
) -> list[dict[str, Any]]:
    base_cfg = load_config(config_path)
    if epochs is not None:
        base_cfg.epochs = epochs
    if batch_size is not None:
        base_cfg.batch_size = batch_size
    if device is not None:
        base_cfg.device = device
    if dae_pretrain_epochs is not None:
        base_cfg.dae_pretrain_epochs = dae_pretrain_epochs
    selected = ABLATION_MATRIX
    if only is not None:
        selected = [item for item in ABLATION_MATRIX if item["name"] == only]
        if not selected:
            valid = ", ".join(item["name"] for item in ABLATION_MATRIX)
            raise ValueError(f"Unknown ablation variant: {only}. Valid variants: {valid}")
    out_path = Path(out_csv)
    out_dir = out_path.parent
    rows = []
    for item in selected:
        rows.append(run_one(item["name"], item["override"], base_cfg=base_cfg, config_path=config_path, out_dir=out_dir, dummy=dummy, seed=seed, num_workers=num_workers, save_attention=item["name"] == "full_6block" or only is not None))
    baseline_auc = rows[0]["macro_auc"] if rows else 0.0
    baseline_f1 = rows[0]["macro_f1"] if rows else 0.0
    for row in rows:
        row["auc_drop"] = float(baseline_auc - row["macro_auc"])
        row["f1_drop"] = float(baseline_f1 - row["macro_f1"])
    write_results_csv(rows, out_path)
    if len(rows) > 1:
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dae-pretrain-epochs", type=int, default=None)
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        dae_pretrain_epochs=args.dae_pretrain_epochs,
    )


if __name__ == "__main__":
    main()
