# A Coherent Multi-Block Deep Architecture for 12-Lead ECG Diagnostic Classification on PTB-XL

## Abstract

This project studies multi-label diagnostic classification of 12-lead electrocardiogram recordings using PTB-XL, a research-paper dataset containing clinical ECG records with structured diagnostic annotations. We propose a coherent deep architecture that combines a denoising autoencoder, one-dimensional convolutional feature extraction, residual learning, recurrent sequence modeling, temporal attention, and a variational autoencoder regularization branch. The design follows the structure of the ECG signal: the autoencoder first encourages noise-robust representations, the convolutional stack extracts local P-QRS-T morphology, recurrent layers model temporal rhythm dependencies, attention emphasizes diagnostically relevant time segments, and the VAE branch regularizes the latent space. The model is evaluated with the official PTB-XL fold protocol using macro-AUC and macro-F1. We also provide an ablation framework that removes or modifies one block at a time, producing result tables and performance-drop visualizations for a conference-paper-style analysis.

## 1. Introduction

Automated ECG interpretation is a suitable problem for deep learning because clinically relevant patterns appear at multiple temporal scales. Short local waveform shapes such as QRS complexes and ST-T changes must be modeled together with longer rhythm-level dependencies. A single shallow classifier is therefore not sufficient for the course objective, which requires a multi-layer model incorporating CNNs, at least one recurrent block, and autoencoder models.

The task is formulated as five-class multi-label classification. Each 10-second, 12-lead ECG is mapped to the PTB-XL diagnostic superclasses NORM, MI, STTC, CD, and HYP. The model outputs five logits and is trained with binary cross-entropy, because a record may contain more than one diagnostic superclass.

The main contributions of the project are as follows. First, we use PTB-XL, a research-paper dataset rather than an overly common educational dataset. Second, we construct a coherent architecture with more than five distinct deep-learning blocks. Third, we implement a repeatable ablation study that quantifies the contribution of each block. Fourth, we document the method in a conference-paper style so that the repository contains both executable code and scientific explanation.

## 2. Dataset

PTB-XL is selected because it is a large-scale ECG dataset introduced in the research literature and distributed with standardized metadata, diagnostic labels, and benchmark folds. The dataset contains 12-lead ECG recordings sampled at 100 Hz or 500 Hz. This project uses the 100 Hz version, giving an input tensor of shape `[12, 1000]` for each 10-second record.

The label pipeline maps SCP diagnostic statements to five PTB-XL diagnostic superclasses: NORM, MI, STTC, CD, and HYP. Since ECG records can have multiple diagnostic categories, the target is a five-dimensional multi-hot vector rather than a single class index.

The project follows the official stratified fold protocol. Folds 1 through 8 are used for training, fold 9 is used for validation and hyperparameter selection, and fold 10 is kept as the final test fold. This split is important because the ablation study must compare all model variants under the same data protocol.

## 3. Method

### 3.1 Denoising Autoencoder

The denoising autoencoder is placed at the beginning of the model. ECG signals can contain baseline wander, powerline interference, motion artifacts, and lead-specific noise. The DAE learns to reconstruct clean signals from corrupted inputs and therefore encourages the encoder to capture stable structure instead of memorizing noise. In the full model, the DAE output is passed to the convolutional feature extractor. In the training pipeline, the DAE can also be pretrained with reconstruction loss before the supervised phase.

### 3.2 One-Dimensional CNN

The CNN block operates directly on the temporal ECG signal. One-dimensional convolution is appropriate because diagnostic morphology is local in time: P waves, QRS complexes, T waves, ST changes, and conduction-related patterns occur in short temporal neighborhoods. Convolutions provide parameter sharing across time and reduce the number of parameters compared with fully connected processing of the raw waveform.

### 3.3 Residual Learning

Residual connections are used inside the CNN feature extractor. The residual path improves gradient flow and makes the convolutional stack easier to optimize. In the ablation study, residual blocks are replaced with plain convolutional blocks while keeping the rest of the architecture fixed. This isolates the effect of the skip connections.

### 3.4 Recurrent Sequence Modeling

After CNN feature extraction, the feature map is interpreted as a shorter temporal sequence and processed with an LSTM or GRU. The recurrent block models dependencies across time, which is important for rhythm-related abnormalities and patterns that cannot be fully described by isolated local morphology. The baseline uses a bidirectional LSTM because the diagnostic decision is made offline from the complete ECG segment. The ablation matrix includes both a GRU replacement and a unidirectional variant.

### 3.5 Temporal Attention

The attention layer pools recurrent outputs into a fixed-length vector. Instead of assigning equal importance to all time steps, attention learns a weight for each temporal position. This is useful for ECG classification because a diagnostic clue may appear only in a short segment. The attention weights also provide interpretability evidence by showing which temporal regions contributed most strongly to the decision.

### 3.6 Variational Autoencoder Branch

The VAE branch is used as a latent regularization component. It encodes the raw ECG into a stochastic latent representation and contributes a KL-divergence term to the total loss. This branch encourages a smoother latent space and provides a distinct autoencoding component from the DAE. In the ablation study, the VAE is removed to measure whether this regularization improves test performance.

### 3.7 Complete Architecture

The full data flow is:

```text
Raw ECG [B, 12, 1000]
    -> Denoising Autoencoder
    -> 1D CNN with residual blocks
    -> BiLSTM or BiGRU
    -> Temporal attention or mean pooling
    -> Classification head
    -> Five diagnostic logits
```

The VAE branch runs in parallel and contributes an auxiliary KL loss during training. This makes the model coherent rather than a random collection of blocks: each component addresses a different property of the ECG classification problem.

## 4. Training Setup

The supervised objective is binary cross-entropy with logits. For a sample with target vector `y` and predicted logits `z`, the BCE term is computed independently for each superclass and averaged across the batch.

The total training loss is:

```text
L = BCE(z, y) + lambda_L1 * ||W||_1 + beta_VAE * KL(q(z_latent|x) || p(z_latent))
```

L2 regularization is implemented through optimizer weight decay. L1 regularization is optionally added as a direct penalty over model weights excluding bias parameters. Dropout and batch normalization are used inside the model. Early stopping monitors validation macro-AUC on fold 9. The test fold is not used during hyperparameter tuning.

The optimizer is selected from the configuration file. The training code supports Adam, RMSProp, and SGD with momentum. Learning-rate scheduling supports cosine decay, linear decay, or no schedule. The default baseline configuration uses Adam with cosine scheduling.

## 5. Evaluation Metrics

The primary metric is macro-AUC because the task is multi-label and class imbalance is expected. Macro-AUC is computed independently per superclass and averaged over valid classes. Macro-F1 is reported as a secondary threshold-based metric. The implementation includes safeguards for classes that contain only one label value in a small dummy or validation batch.

## 6. Ablation Study

The ablation study is the main responsibility of Person D. The purpose is to show whether each architectural block contributes to the final result. All variants are trained and evaluated with the same fold protocol, same metrics, and same baseline configuration except for the controlled override being tested.

The implemented ablation matrix is:

| Variant | Modification |
|---|---|
| `full_6block` | Complete baseline model |
| `no_dae` | Removes the denoising autoencoder |
| `no_residual` | Replaces residual CNN blocks with plain convolutional blocks |
| `no_attention` | Replaces temporal attention with mean pooling |
| `no_vae` | Removes the VAE regularization branch |
| `gru_instead_lstm` | Replaces LSTM with GRU |
| `unidirectional` | Uses a unidirectional recurrent layer instead of a bidirectional one |
| `minimal_supported` | Removes DAE, residual connections, attention, and VAE while keeping the assignment-required CNN and recurrent pathway |

The script `ablation/run_ablation.py` produces the following artifacts:

| Artifact | Purpose |
|---|---|
| `ablation/results/ablation_summary.csv` | Main result table with macro-AUC, macro-F1, and performance drops |
| `ablation/results/auc_drop.png` | Bar chart showing macro-AUC decrease relative to the full model |
| `ablation/results/f1_drop.png` | Bar chart showing macro-F1 decrease relative to the full model |
| `ablation/results/attention_example.png` | Example attention-weight visualization |
| `ablation/results/history_<variant>.csv` | Per-epoch training and validation history for each variant |
| `ablation/results/config_<variant>.json` | Exact configuration used for each variant |

### Ablation Results Placeholder

| Variant | Macro-AUC | Macro-F1 | AUC Drop | F1 Drop | Interpretation |
|---|---|---|---|---|---|
| `full_6block` | | | | | |
| `no_dae` | | | | | |
| `no_residual` | | | | | |
| `no_attention` | | | | | |
| `no_vae` | | | | | |
| `gru_instead_lstm` | | | | | |
| `unidirectional` | | | | | |
| `minimal_supported` | | | | | |

## 7. Expected Result Interpretation

After running the real PTB-XL experiment, the result table should be interpreted by comparing each variant with `full_6block`. A positive AUC drop means the removed component improved performance in the full model. A small or negative drop means the component did not help under the current hyperparameter setting and should be discussed honestly.

The attention ablation tests whether learned temporal weighting is better than uniform mean pooling. The residual ablation tests whether skip connections improve optimization. The DAE and VAE ablations test whether reconstruction-based representation learning and latent regularization add value beyond the supervised CNN-RNN classifier. The GRU and unidirectional variants test recurrent design choices.

## 8. Limitations

The ablation results are meaningful only after the full data pipeline, model blocks, and trainer are integrated correctly. Dummy-mode results are useful only for verifying that the experiment runner works; they must not be reported as final scientific performance. The real paper table should be filled from the CSV generated on the PTB-XL folds.

## 9. Conclusion

The project satisfies the course requirement by combining CNN, recurrent, and autoencoder-based components in a single coherent model. The architecture is designed around the structure of ECG signals and is evaluated with a repeatable ablation framework. The final scientific claim should be based on the test-fold macro-AUC and macro-F1 results generated by the completed ablation pipeline.

## References

Wagner, P., Strodthoff, N., Bousseljot, R.-D., et al. PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 2020.

Strodthoff, N., Wagner, P., Schaeffter, T., and Samek, W. Deep learning for ECG analysis: Benchmarks and insights from PTB-XL. IEEE Journal of Biomedical and Health Informatics, 2021.

Goodfellow, I., Bengio, Y., and Courville, A. Deep Learning. MIT Press, 2016.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. CVPR, 2016.

Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural Computation, 1997.

Cho, K., van Merriënboer, B., Gulcehre, C., et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP, 2014.

Bahdanau, D., Cho, K., and Bengio, Y. Neural machine translation by jointly learning to align and translate. ICLR, 2015.

Kingma, D. P. and Welling, M. Auto-encoding variational Bayes. ICLR, 2014.

Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P.-A. Extracting and composing robust features with denoising autoencoders. ICML, 2008.
