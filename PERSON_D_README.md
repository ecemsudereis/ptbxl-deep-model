# Ablation Study Notes

These notes cover the ablation experiments I ran for the PTB-XL ECG classification project. The goal was simple: train the full model, remove or replace one component at a time, and check how each change affects macro-AUC and macro-F1 on the held-out test fold.

## What the ablation script does

The ablation runner is located at:

```text
ablation/run_ablation.py
```

It trains the selected model variant, evaluates it on the test split, and writes the results under:

```text
ablation/results/
```

The main output file is:

```text
ablation/results/ablation_summary.csv
```

The script also saves per-variant training logs, the exact configuration used for each run, and the final comparison plots.

## Smoke test

Before running the full PTB-XL experiment, I used a short dummy run to make sure the script worked end to end:

```bash
python -m ablation.run_ablation --dummy --epochs 1 --batch-size 16 --device cpu --dae-pretrain-epochs 0 --only no_attention
```

This only checks whether the pipeline runs. It is not a real experiment and the numbers from this command are not used in the report.

## Final experiment command

For the final run, I used the real PTB-XL data with the following command:

```bash
python -m ablation.run_ablation --config configs/baseline.yaml --device cuda --epochs 25 --batch-size 16 --dae-pretrain-epochs 0 --patience 6 --num-workers 0
```

This was run on an RTX 3060 6GB GPU. The batch size was kept at 16 to avoid GPU memory issues. If the same command fails on another machine because of memory, batch size 8 is the safer option.

## Ablation variants

The final experiment includes eight variants:

```text
full_6block
no_dae
no_residual
no_attention
no_vae
gru_instead_lstm
unidirectional
minimal_supported
```

The `minimal_supported` variant is not a pure CNN-only model. It keeps the required CNN and recurrent path active, while disabling the optional DAE, residual, attention, and VAE components.

## Final outputs

After the full run, the results folder contains:

```text
ablation/results/ablation_summary.csv
ablation/results/auc_drop.png
ablation/results/f1_drop.png
ablation/results/attention_example.png
ablation/results/history_<variant>.csv
ablation/results/config_<variant>.json
```

The final CSV must contain one row for each of the eight variants. A one-row file means that only a partial test was run.

## Final result summary

| Variant | Macro-AUC | Macro-F1 | AUC Drop | F1 Drop | Best Epoch |
|---|---:|---:|---:|---:|---:|
| full_6block | 0.8733 | 0.5807 | 0.0000 | 0.0000 | 23 |
| no_dae | 0.9074 | 0.6763 | -0.0341 | -0.0957 | 17 |
| no_residual | 0.8383 | 0.5113 | 0.0350 | 0.0694 | 23 |
| no_attention | 0.8994 | 0.6581 | -0.0261 | -0.0774 | 16 |
| no_vae | 0.8704 | 0.5712 | 0.0029 | 0.0094 | 21 |
| gru_instead_lstm | 0.8933 | 0.6358 | -0.0200 | -0.0551 | 16 |
| unidirectional | 0.8927 | 0.6345 | -0.0194 | -0.0539 | 24 |
| minimal_supported | 0.8998 | 0.6854 | -0.0265 | -0.1047 | 21 |

The full model is not the best-performing variant in this run. The strongest result comes from `no_dae` in macro-AUC and from `minimal_supported` in macro-F1. The clearest performance loss appears in `no_residual`, which suggests that the residual CNN path is the most useful component among the tested blocks.

Negative drop values mean that the ablated variant performed better than the full model. This is not a mistake; it shows that adding more blocks does not automatically improve generalization.

## Files to commit

For the final repository, I include the ablation code, the paper text, the final CSV table, the comparison plots, the attention plot, and the run histories.

I do not commit raw PTB-XL records, processed NumPy arrays, virtual environments, cache files, or model checkpoint files.
