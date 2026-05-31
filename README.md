# Audio Emotion Classification — SVM & CNN

A machine learning pipeline for classifying 7 human emotions from raw audio files, built for my Final Project in ELEC 378.
See [ELEC 378 Audio Emotion Classification Kaggle Competition](https://www.kaggle.com/competitions/elec-378-sp-2025-audio-emotion-classification) for more details.

**Final Kaggle scores:** 72.535% accuracy on public dataset / 71.853% accuracy on complete dataset, including unknown data

*Team: Anthony Zhang, Ankit Burudgunte, John David Villarreal*

---

## Problem

Classify `.wav` audio clips into one of 7 emotions: **Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised**. The dataset contains ~13,000 labeled training clips and a held-out test set evaluated on Kaggle.

---

## Pipeline Overview

```
Raw .wav files
    ↓
Preprocessing (silence removal, Butterworth bandpass filter 20Hz–20kHz)
    ↓
Feature Extraction (librosa — MFCCs, spectral features, chroma, RMS, ZCR)
    ↓
Feature Selection (Recursive Feature Elimination via Random Forest, 183 → 100 dims)
    ↓
Dimensionality Reduction (PCA, 99.8% variance retained)
    ↓
Model Training (SVM with RBF kernel / CNN on Mel spectrograms)
    ↓
5-Fold Stratified Cross-Validation
    ↓
Kaggle Submission
```

---

## Feature Extraction

Features were selected based on research into speech emotion recognition, then iteratively tested for impact on accuracy. The final feature vector (pre-RFE) has 183 dimensions, reduced to 100 via RFE.

### MFCC Features
- **MFCCs** (20 coefficients) — Mel-Frequency Cepstral Coefficients capture the short-term power spectrum on the mel scale, mimicking human auditory perception. Mean, std, min, max, and median are extracted per coefficient.
- **Deltas** — First-order difference of MFCCs across time frames (approximates 1st derivative), capturing how the spectral envelope changes.
- **Delta-Deltas** — Second-order difference (approximates 2nd derivative), capturing acceleration of spectral change.

### Spectral Features
- **Spectral Contrast** — Amplitude difference between peaks and valleys in frequency sub-bands; distinguishes harmonic vs. noisy sounds.
- **Spectral Centroid** — Weighted average frequency ("center of mass" of the spectrum).
- **Spectral Bandwidth** — Spread of energy around the centroid; wider = more distributed energy.
- **Spectral Flatness** — How noise-like vs. tonal a sound is (1 = noise, 0 = pure tone); identifies whispering, yelling, etc.
- **Spectral Rolloff** — Frequency below which 85% of total signal energy resides.

### Other Features
- **Chroma** (12 pitch classes) — Maps frequencies to musical notes (C, C#, D, …); captures tonal/harmonic qualities affected by emotion.
- **RMS** — Root mean square energy; measures average loudness/intensity over time.
- **Zero-Crossing Rate (ZCR)** — Rate at which the signal crosses zero; distinguishes voiced vs. unvoiced sounds.

Features excluded after testing: pitch (high compute cost, marginal accuracy gain).

### Preprocessing
- **Silence removal**: `librosa.effects.split` with `top_db=20`
- **Butterworth bandpass filter**: 4th-order, 20 Hz–20 kHz, applied before feature extraction
- **Aggregation**: Mean, std, min, max, median per feature → fixed-length vector regardless of audio duration
- **Scaling**: `StandardScaler` (zero mean, unit variance)

---

## Models

### Support Vector Machine (SVM) — *Best model*

An RBF kernel SVM trained on the 100-dimensional RFE-reduced, PCA-projected feature vectors.

**Hyperparameter tuning** via `GridSearchCV` with 5-fold stratified cross-validation:

| Hyperparameter | Values tested | Best |
|---|---|---|
| Kernel | linear, poly, rbf, sigmoid | rbf |
| C | 0.1, 1, 10, 100 | 10 |
| Gamma | scale, auto | scale |

**Overfitting mitigation:**
- Recursive Feature Elimination (RFE) with Random Forest to reduce 183 → 100 features
- PCA retaining 99.8% of variance
- 5-fold stratified cross-validation (average train accuracy ~70%)

**Results:** 72.535% public / 71.853% private — the near-identical scores confirm the model generalized well without overfitting to the training set.

RFE selected: MFCCs, RMS, spectral contrast, ZCR, spectral bandwidth, spectral flatness, spectral rolloff. Removed: chroma, deltas, spectral centroid.

---

### Convolutional Neural Network (CNN)

A CNN trained on 64×64 log-Mel spectrograms. Audio silence is trimmed before spectrogram generation; spectrograms are resized to fixed 64×64 via `AdaptiveAvgPool2d`.

**Architecture:**

```
Input: (1, 64, 64) — log-Mel spectrogram

Conv2d(1→32, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2)
Conv2d(32→32, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2)
Conv2d(32→16, 3×3) → ReLU → BatchNorm2d → MaxPool2d(2)
AdaptiveAvgPool2d(1,1) → Flatten

Linear(16→16) → ReLU → Dropout(0.5)
Linear(16→7)   ← output logits for 7 emotions

Total parameters: 14,743
```

**Training:** Adam optimizer (lr=0.003), CrossEntropyLoss, 50 epochs, batch size 8.

**Overfitting mitigation:** BatchNorm after each conv layer, MaxPool for dimensionality reduction, Dropout(0.5) before output, mini-batch training.

**Results:** ~60% training accuracy, 62.47% Kaggle — the CNN overfits: one run reached 85% train accuracy but only 62% on the test set. Long training times (1–3 hours per run) limited hyperparameter exploration.

---

## Results Summary

| Model | Train Accuracy (CV) | Kaggle Public | Kaggle Private |
|---|---|---|---|
| SVM (RBF, C=10) | ~70% | **72.54%** | **71.85%** |
| CNN (50 epochs) | ~60% | 62.47% | — |

The SVM outperformed the CNN primarily due to better-engineered features and more tractable hyperparameter tuning. Key takeaway: feature extraction quality had a larger impact on accuracy than model complexity or hyperparameter choices.

---

## Running the Notebook

The notebook (`audio_emotion_classification.ipynb`) was developed in **Google Colab** with the dataset stored in Google Drive.

**Dependencies:**
```
librosa
numpy
scipy
scikit-learn
torch
matplotlib
```

Install with:
```bash
pip install librosa numpy scipy scikit-learn torch matplotlib
```

Update the `file_path` variables to point to your local dataset directory structure:
```
dataset/
  Angry/    *.wav
  Disgusted/ *.wav
  Fearful/  *.wav
  Happy/    *.wav
  Neutral/  *.wav
  Sad/      *.wav
  Suprised/ *.wav
  Test/     1.wav, 2.wav, ...
```

---

## References

1. Feature Extraction for Speech Emotion Recognition — [MDPI Sensors 2024](https://www.mdpi.com/1424-8220/24/17/5797)
2. MFCCs — [Intuitive Understanding of MFCCs](https://medium.com/@derutycsl/intuitive-understanding-of-mfccs-836d36a1f779)
3. Librosa feature extraction — [librosa docs](https://librosa.org/doc/0.10.1/feature.html)
4. Deltas — [Aalto Speech Processing Book](https://speechprocessingbook.aalto.fi/Representations/Deltas_and_Delta-deltas.html)
5. Random Forest — [Wikipedia](https://en.wikipedia.org/wiki/Random_forest)
