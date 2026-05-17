# INTERFACE CONTRACT

> This file is the **fixed foundation** of the project. Nothing here
> changes without team approval. Everyone works against this contract,
> without waiting for one another.

---

## 1. Tensor Shapes

| Stage | Shape | Description |
|---|---|---|
| Raw input (model) | `(B, 12, 1000)` | 12 leads × (100 Hz × 10 s = 1000 samples), float32 |
| DAE output | `(B, 12, 1000)` | denoised signal (reconstruction) |
| DAE latent | `(B, D_dae)` | compressed representation (D_dae in config) |
| CNN+Residual output | `(B, C, L')` | C channels, L' shortened time axis |
| Recurrent output | `(B, L', H)` | H = hidden size (2×hidden if bidirectional) |
| Attention output | `(B, H)` | vector pooled over the time axis |
| Classifier output | `(B, 5)` | logits — sigmoid + BCEWithLogitsLoss |

`B` = batch size. All blocks use the `batch_first=True` convention.

---

## 2. Labels — 5 Superclasses (Multi-label)

| Index | Code | Meaning |
|---|---|---|
| 0 | `NORM` | Normal ECG |
| 1 | `MI`   | Myocardial infarction |
| 2 | `STTC` | ST/T change |
| 3 | `CD`   | Conduction disturbance |
| 4 | `HYP`  | Hypertrophy |

A record may belong to **more than one** class → multi-label.
Target tensor: `(B, 5)` float, 0/1. Loss: `nn.BCEWithLogitsLoss`.

---

## 3. Fold Split (PTB-XL Official Standard)

PTB-XL assigns each record a `strat_fold` (1–10). Benchmark standard:

| Fold | Use |
|---|---|
| 1 – 8 | Train |
| 9 | Validation (hyperparameter tuning, early stopping) |
| 10 | Test (final only — used once) |

This split is **what makes the ablation scientific**: all variants are
compared on the same folds.

---

## 4. Metrics

- **Primary:** macro-AUC (mean of per-class ROC-AUC)
- **Secondary:** macro-F1 (per class, threshold 0.5)

Both with a single signature in `training/metrics.py`:
`compute_metrics(y_true: Tensor[B,5], y_logits: Tensor[B,5]) -> dict`

---

## 5. Config Keys (summary)

Full list in `configs/baseline.yaml`. The critical ablation flags live
in `models/full_model.py` `FullModel.__init__`:

```
use_dae: bool          # Denoising AE pretraining/branch on?
use_residual: bool     # residual connections in the CNN on?
rnn_type: "lstm"|"gru" # recurrent block type
bidirectional: bool    # bidirectional or not
use_attention: bool    # attention pooling on?
use_vae: bool          # VAE latent regularization on?
```

These flags are exactly the axes of the ablation matrix.
