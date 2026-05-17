# A Coherent Multi-Block Deep Architecture for 12-Lead ECG Diagnostic Classification on PTB-XL

> **Conference-paper-style write-up (bonus +15).** This skeleton is
> designed around the instructor's emphasis on theory: every
> architectural choice is tied to a Goodfellow concept covered in
> class. Person D owns it; everyone contributes. Format: short
> conference paper (e.g. ~6-8 page equivalent).

---

## Abstract

*(~150 words — written last.)* Problem, PTB-XL choice, the 6-block
coherent architecture, the ablation finding, and the comparison to the
benchmark, in one paragraph.

## 1. Introduction

- Importance of ECG diagnosis and the motivation for automated
  classification.
- Problem definition: 12-lead, 5 superclasses, multi-label.
- Our contributions (bullet list): (i) coherent 6-block architecture,
  (ii) comprehensive ablation, (iii) comparison to the PTB-XL benchmark.

## 2. Related Work

- The PTB-XL dataset (Wagner et al. 2020) and why it was chosen.
- PTB-XL benchmark results (Strodthoff et al. 2021) — comparison basis.
- Brief survey of CNN/RNN/AE approaches on ECG.

## 3. Dataset

- PTB-XL: ~21.8k records, 100 Hz, 12 leads, 10 s.
- SCP → 5 superclass mapping (NORM, MI, STTC, CD, HYP).
- Official strat_fold split (1-8 / 9 / 10) — why required for a
  scientific comparison.
- Preprocessing: filtering + per-lead normalization.

## 4. Method — Architecture and Justifications

> Each subsection: block + **justification with the concept taught in
> class**.

### 4.1 Denoising Autoencoder (W8)
Noisy ECG → manifold learning; avoiding the identity trap.
*Justify:* Goodfellow Ch. 14.5, Vincent et al. 2008.

### 4.2 1D CNN + Residual (W6, W5)
Local P-QRS-T morphology; sparse connectivity, parameter sharing,
translation equivariance. Residual against vanishing gradients.
*Justify:* W6, He et al. 2016.

### 4.3 BiLSTM/BiGRU (W7)
Temporal/rhythm dependency; long-term dependency via gating; offline →
bidirectional is justified. *Justify:* W7, Hochreiter & Schmidhuber
1997.

### 4.4 Temporal Attention (W7/W8)
Focus on diagnostic segments + interpretability. *Justify:*
content-based memory access, Bahdanau et al. 2015.

### 4.5 VAE (W8)
Latent regularization + minority-class augmentation; a distinct block
from the DAE. *Justify:* stochastic encoder/decoder, Kingma & Welling
2014.

### 4.6 Fusion (Coherence)
Data-flow diagram (the diagram in the README) — why the blocks are in
this order and why they form a coherent whole.

## 5. Training Setup

> This section directly addresses the assignment's "detailed
> hyperparameter tuning and regularization" requirement.

- **Optimizer & LR (W5):** Adam vs RMSProp, LR schedule
  (cosine/linear).
- **Hyperparameter tuning (W2):** search on the validation fold (9);
  test fold (10) only for the final. Searched: lr, hidden, dropout,
  vae_beta.
- **Regularization (W4):** L2 weight decay, L1, dropout, BatchNorm,
  early stopping (patience), label smoothing — which one and why.
- Loss: BCE + λ·L1 + β·KL; optional DAE pretraining phase.

## 6. Experiments & Results

- Main result: full-model macro-AUC / macro-F1 (test fold).
- Comparison table vs the PTB-XL benchmark (Strodthoff 2021).

## 7. Ablation Study (bonus +15)

- Table 2: the 8 variants from ABLATION_MATRIX × {macro-AUC, macro-F1}.
- Figure: performance drop when each block is removed (bar chart).
- Discussion: how much each block contributes; LSTM↔GRU,
  bidir↔unidir comparison.
- Attention-weight visualization (interpretability evidence).

## 8. Conclusion

Summary of findings; the benefit of the coherent multi-block design;
limitations and possible future work.

## References

See `references.bib`.

---

### Writing note (for the team)
- Every "Justify" sentence must tie to a course week + a citation.
- Result tables should be auto-fed from `ablation/results/` output.
- The architecture diagram (from the README ASCII flow) will be turned
  into a professional figure.
