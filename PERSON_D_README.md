# Ablation Study Notes

This file summarizes how I ran the ablation part of the PTB-XL project and where the outputs are saved.

## Purpose

The goal of the ablation study is to check how much each architectural block contributes to the final ECG classifier. The full model is compared with reduced versions where one block is removed or replaced. The main metrics are macro-AUC and macro-F1.

## Quick smoke test

I first use dummy data to make sure the script, model call, training loop, metric calculation, and CSV export work end to end.

```bash
python -m ablation.run_ablation --dummy --epochs 1 --batch-size 16 --device cpu --dae-pretrain-epochs 0 --only no_attention
```

This command runs only the `no_attention` variant and writes the result to:

```text
ablation/results/ablation_summary.csv
```

The dummy run is only a pipeline check. I do not use dummy metrics as final experimental results.

## Full dummy ablation run

```bash
python -m ablation.run_ablation --dummy --epochs 1 --batch-size 16 --device cpu --dae-pretrain-epochs 0
```

This runs all ablation variants with synthetic data. I use it as a final sanity check before running the real PTB-XL experiments.

## Final PTB-XL run

The real experiments require the PTB-XL files to be available under the expected data directory. If the dataset has not been downloaded yet, I run:

```bash
python -m data.download
```

Then I start the final ablation run with:

```bash
python -m ablation.run_ablation --config configs/baseline.yaml
```

## Output files

The ablation script saves its outputs under `ablation/results/`.

```text
ablation/results/ablation_summary.csv
ablation/results/auc_drop.png
ablation/results/f1_drop.png
ablation/results/attention_example.png
ablation/results/history_<variant>.csv
ablation/results/config_<variant>.json
```

`ablation_summary.csv` is the main table for the report. The AUC and F1 plots show how much each reduced model drops compared with the full model.

## Ablation variants

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

The variants compare the full architecture against versions without the denoising autoencoder, residual block, attention module, VAE branch, bidirectional recurrence, or LSTM recurrence. The GRU variant checks whether a simpler recurrent unit can replace LSTM.

`minimal_supported` keeps the main CNN and recurrent path active while disabling the optional DAE, residual, attention, and VAE components. I used this name instead of calling it a pure CNN model, because the current classifier interface still keeps the recurrent path.

## Notes for reporting

The paper should use the real PTB-XL results, not the dummy results. Dummy runs only show that the code executes correctly. After the final run, I transfer the values from `ablation_summary.csv` into the ablation table in `paper/paper.md`.

For GitHub, I keep source code, documentation, the final summary CSV, and final plots. I do not commit the raw PTB-XL records or large generated model files.
