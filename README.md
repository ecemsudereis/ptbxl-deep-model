# PTB-XL Deep Model — Multi-Block Deep Learning Project

Deep Learning course term project · 4-person team · PyTorch

A multi-layer deep model that **combines 6 distinct architectural blocks
in a coherent manner** for diagnostic classification of 12-lead ECG
signals. The dataset is sourced from a research paper (PTB-XL).

---

## 1. Project Summary

| | |
|---|---|
| **Task** | 12-lead ECG → 5 superclass multi-label diagnostic classification |
| **Dataset** | PTB-XL (Wagner et al., *Scientific Data* 2020, PhysioNet) |
| **Framework** | PyTorch (the course is PyTorch-based) |
| **Blocks** | Denoising AE · 1D CNN · Residual · BiLSTM/BiGRU · Attention · VAE |
| **Evaluation** | macro-AUC (primary), macro-F1 · official PTB-XL folds |

### Bonus targets (total +60)

- Research-paper dataset → **+15** (PTB-XL)
- 5+ distinct blocks → **+15** (we build 6 blocks)
- Ablation study → **+15** (contribution of each block measured)
- Conference-paper-style write-up in the GitHub repo → **+15** (`paper/`)

---

## 2. ⚠️ READ THIS FIRST: The Interface Contract

So that everyone can work **in parallel from day one**, all tensor shapes
are fixed up front. Details → [`CONTRACT.md`](CONTRACT.md). Summary:

```
Input  : (batch, 12, 1000)   # 12 leads, 10 s at 100 Hz = 1000 samples
Output : (batch, 5)          # NORM, MI, STTC, CD, HYP — sigmoid + BCE
Folds  : 1–8 train · 9 validation · 10 test   (PTB-XL benchmark standard)
Metric : macro-AUC (primary) + macro-F1
```

Thanks to this contract:
- Whoever writes the architecture can test with **dummy tensors**
  without waiting for the real data.
- Whoever writes the training stack can test with a **dummy model**
  without waiting for the real architecture.
- Nobody waits for anybody.

---

## 3. Task Division (4 People)

| Person | Owned area | Folder | Week-1 deliverable |
|---|---|---|---|
| **A — Data** | Download, WFDB reading, filtering/normalization, SCP→5 superclass mapping, fold split, `Dataset`/`DataLoader` | `data/` | Working pipeline + example signal plots |
| **B — Architecture** | 6 modular `nn.Module` blocks, fused in the main model, ablation flags | `models/` | Each block passes its dummy-tensor unit test |
| **C — Training** | Training loop, optimizer (Adam/RMSProp – W5), LR schedule, regularization (L1/L2, dropout, BatchNorm, early stopping, label smoothing – W4), metrics, config system | `training/`, `configs/` | End-to-end run with dummy model+data |
| **D — Ablation + Paper** | Ablation experiment matrix, result tables/plots, conference paper, repo organization | `ablation/`, `paper/` | Paper skeleton + ablation experiment plan |

> **Repo owner (you):** Integration coordination is yours. Track branch
> merges and the `models/full_model.py` ↔ `data/dataset.py` ↔
> `training/trainer.py` junction. You can also own role A or C.

---

## 4. Timeline

| Week | What happens |
|---|---|
| **Week 1** | 4 people work in parallel — each in their own folder, per the contract |
| **Week 2** | **Integration point:** A+B+C merge → first real baseline run |
| **Week 3** | D runs the ablations + paper is written (full system must be up) |

---

## 5. Repo Structure

```
ptbxl-deep-model/
├── README.md              # this file
├── CONTRACT.md            # interface contract — ASK THE TEAM BEFORE CHANGING
├── requirements.txt
├── .gitignore
├── configs/               # C: hyperparameter configs (yaml)
│   └── baseline.yaml
├── data/                  # A
│   ├── download.py        # PTB-XL download
│   ├── preprocess.py      # filtering, normalization, SCP→superclass
│   └── dataset.py         # PyTorch Dataset + DataLoader
├── models/                # B — one block per file
│   ├── dae.py             # Denoising Autoencoder
│   ├── cnn_residual.py    # 1D CNN + Residual blocks
│   ├── recurrent.py       # BiLSTM / BiGRU
│   ├── attention.py       # Temporal attention
│   ├── vae.py             # Variational Autoencoder
│   └── full_model.py      # Main model fusing all (ablation flags)
├── training/              # C
│   ├── trainer.py         # training loop + regularization + optimizer
│   └── metrics.py         # macro-AUC, macro-F1
├── ablation/              # D
│   ├── run_ablation.py    # ablation runner
│   └── results/           # output tables/plots
├── paper/                 # D
│   ├── paper.md           # conference-paper skeleton
│   └── references.bib     # citations (PTB-XL, benchmark, Goodfellow...)
└── notebooks/             # exploration & visualization
```

Each folder maps to one person → minimal merge conflicts.

---

## 6. Setup

```bash
git clone <repo-url>
cd ptbxl-deep-model
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Every module has its own dummy-tensor smoke test:

```bash
python -m models.cnn_residual    # B: block shape check
python -m training.trainer       # C: dummy end-to-end run
```

---

## 7. Working Rules

1. **`CONTRACT.md` is sacred.** Any shape/fold/metric change = ask the team first.
2. Everyone works on a `feature/<name>-<topic>` branch in their own folder.
3. Pass your module's smoke test before opening a PR.
4. The raw PTB-XL data under `data/` is in `.gitignore` — it is **not** committed.
5. Commit messages short and in English: `feat(data): add SCP to superclass mapping`.

---

## 8. Citations

Core sources for the paper and justifications are in `paper/references.bib`.
Core: PTB-XL (Wagner 2020), PTB-XL benchmark (Strodthoff 2021),
Deep Learning textbook (Goodfellow, Bengio, Courville 2016).
