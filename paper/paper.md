# A Multi-Block Deep Learning Model for PTB-XL ECG Classification

## Abstract

This project focuses on multi-label diagnostic classification of 12-lead ECG recordings from the PTB-XL dataset. The model combines several deep learning blocks covered in the course: a denoising autoencoder, one-dimensional convolutional layers, residual connections, recurrent layers, temporal attention, and a variational autoencoder branch. The task is evaluated on the official PTB-XL fold split using macro-AUC and macro-F1. I also ran an ablation study with eight model variants to see which parts of the architecture actually helped the final classifier. The results show that the residual CNN path was important, while some of the heavier auxiliary blocks did not improve generalization under the current training setup.

## 1. Introduction

ECG classification is a good fit for a multi-block neural network because the signal contains information at different time scales. Some findings are local, such as QRS morphology or ST-T changes. Others depend on a longer temporal context, such as rhythm-related patterns. For this reason, the model uses convolutional layers for local waveform features and recurrent layers for temporal dependencies.

The task is formulated as five-label diagnostic classification. Each ECG record can belong to more than one PTB-XL superclass, so the model outputs five logits and is trained with binary cross-entropy rather than a single softmax class.

The project also avoids using an overly common dataset such as MNIST or Fashion-MNIST. PTB-XL is a research dataset with clinical ECG recordings, structured metadata, and a standard benchmark split, which makes it more suitable for this assignment.

## 2. Dataset

PTB-XL is a large public ECG dataset introduced by Wagner et al. It contains 12-lead ECG recordings with diagnostic annotations. This project uses the 100 Hz signal version, so each input example has the shape:

```text
[12, 1000]
```

This corresponds to 12 leads and 10 seconds of signal sampled at 100 Hz.

The SCP diagnostic statements are mapped into five diagnostic superclasses:

```text
NORM, MI, STTC, CD, HYP
```

The target is represented as a five-dimensional multi-hot vector. The official PTB-XL fold split is used: folds 1-8 for training, fold 9 for validation, and fold 10 for testing.

## 3. Model Architecture

The full model uses six main blocks:

```text
Raw ECG
  -> Denoising Autoencoder
  -> 1D CNN Feature Extractor
  -> Residual CNN Blocks
  -> Bidirectional LSTM
  -> Temporal Attention
  -> Classification Head
```

A VAE branch is also attached as an auxiliary latent regularization component.

### Denoising Autoencoder

The denoising autoencoder is included to encourage noise-tolerant signal representations. ECG recordings can contain baseline drift, powerline noise, motion artifacts, and lead-specific noise. The DAE is intended to make the downstream classifier less sensitive to these effects.

### 1D CNN

The convolutional block extracts local ECG morphology. This is useful because many clinically relevant patterns appear in short waveform segments, including P waves, QRS complexes, T waves, and ST changes.

### Residual Connections

Residual connections are used inside the CNN feature extractor. They help the convolutional stack train more reliably by improving gradient flow. The ablation results show that removing this part caused the largest performance drop.

### Recurrent Layer

The recurrent block processes the CNN feature map as a temporal sequence. The baseline uses a bidirectional LSTM. The ablation study also tests a GRU replacement and a unidirectional recurrent layer.

### Temporal Attention

The attention layer learns a weighted pooling over temporal features instead of averaging all time steps equally. This is useful when only a short part of the ECG contains the most relevant diagnostic signal.

### VAE Branch

The VAE branch adds a latent regularization term. It is included as an additional autoencoder-style component and gives the model a smoother latent representation during training.

## 4. Training Setup

The model is trained with binary cross-entropy with logits. The total loss can also include L1 regularization and VAE KL-divergence, depending on the active configuration.

The final ablation run used:

```text
epochs: 25
batch size: 16
optimizer: Adam
learning-rate schedule: cosine
early stopping patience: 6
device: CUDA
DAE pretraining epochs: 0
```

The final run was performed on an RTX 3060 6GB GPU. Batch size 16 was used because it fit in memory while still keeping the run practical.

## 5. Evaluation Metrics

The main metric is macro-AUC. This is appropriate because the task is multi-label and the diagnostic classes are imbalanced. Macro-F1 is also reported as a secondary metric. The test fold is only used after model selection on the validation fold.

## 6. Ablation Study

The ablation study compares the full model against seven modified versions. Each variant is trained with the same data split and training protocol.

| Variant | Change |
|---|---|
| full_6block | Complete model |
| no_dae | Removes the denoising autoencoder |
| no_residual | Removes residual CNN connections |
| no_attention | Replaces attention with mean pooling |
| no_vae | Removes the VAE branch |
| gru_instead_lstm | Uses GRU instead of LSTM |
| unidirectional | Uses a unidirectional recurrent layer |
| minimal_supported | Keeps the CNN and recurrent path, disables DAE, residual, attention, and VAE |

## 7. Results

The final ablation results on the PTB-XL test fold are shown below.

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

The best macro-AUC is obtained by `no_dae`, while the best macro-F1 is obtained by `minimal_supported`. This means the full six-block model is not automatically the best model. The result is still useful because it shows that extra architectural blocks can add unnecessary complexity if they are not tuned carefully.

The most important negative result is `no_residual`. Removing residual connections lowers macro-AUC from 0.8733 to 0.8383 and macro-F1 from 0.5807 to 0.5113. This supports keeping residual connections in the CNN feature extractor.

Removing the VAE causes only a small drop, so the VAE branch has limited effect in this run. Replacing LSTM with GRU and using a unidirectional recurrent layer both perform better than the full baseline in this setup, which suggests that the baseline recurrent configuration may be larger than necessary.

Negative drop values mean that the ablated model performed better than the full model. This is not treated as an error; it is part of the ablation finding.

## 8. Attention Visualization

The attention plot shows the learned temporal attention weights for an example ECG representation. The weights are not uniform, which means the model gives more importance to selected temporal feature positions. This provides a simple interpretability check for the attention block.

## 9. Discussion

The ablation study gives a more realistic view of the model than reporting only the full architecture. The full model satisfies the course requirement by combining CNN, recurrent, and autoencoder-based components, but the results show that a larger model is not always better. In this run, the DAE and attention modules did not improve test performance, while residual learning was clearly useful.

The stronger performance of simpler variants may be caused by over-parameterization, limited tuning for each added block, or the fact that the preprocessing pipeline already removes part of the signal noise that the DAE was expected to handle. The results suggest that future work should tune the auxiliary losses and pretraining setup more carefully instead of assuming that all blocks should be active at the same time.

## 10. Limitations

The ablation study was run with one seed and one main training configuration. Each variant was not separately hyperparameter-tuned. Because of this, the results should be interpreted as a controlled course-project ablation, not as a final benchmark claim against published PTB-XL methods.

The experiment also uses 25 epochs to keep training practical on the available GPU. Longer training or repeated runs could give more stable estimates.

## 11. Conclusion

The project builds and evaluates a multi-block ECG classifier on PTB-XL. The full architecture includes CNN, recurrent, attention, residual, DAE, and VAE components. The ablation study shows that residual learning is the most clearly useful component in this setup, while some auxiliary blocks can reduce generalization when added without enough tuning. The final results support the main goal of the project: not only to combine the required deep-learning blocks, but also to measure their effect through a repeatable ablation pipeline.

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
