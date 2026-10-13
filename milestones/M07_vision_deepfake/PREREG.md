# Milestone M07 Pre-Registration — Visual Deepfake Detection

**Milestone:** M07 — Visual Deepfake Detection (FaceForensics++, Celeb-DF v2, Forensics, Table XI Replacement)  
**Author:** Antigravity Autonomous Research Agent  
**Date:** 2026-10-13  
**Status:** FROZEN BEFORE EXECUTION  
**Target Ledger IDs:** C013, C015 (CNN baseline row)  

---

## 1. Operational Definitions & Problem Formulation

The visual deepfake detection module operationalizes Section X and Appendix A of the manuscript. Given an input video sequence $V = \{f_1, f_2, \dots, f_T\}$ sampled at 1–2 frames per second (FPS), the vision pipeline:
1. Detects facial bounding boxes and extracts aligned face crops $x_t \in \mathbb{R}^{3 \times 224 \times 224}$.
2. Evaluates frame-level deepfake probability $p_t = \sigma(f_\theta(x_t)) \in [0, 1]$ using a deep convolutional backbone $f_\theta$.
3. Aggregates frame probabilities into video-level prediction score:
   $$S(V) = \frac{1}{T} \sum_{t=1}^T p_t$$
   with an alternative top-$k$ entropy pooling operator (App. D):
   $$S_{\text{entropy}}(V) = \frac{1}{k} \sum_{j \in \text{top-}k} p_j, \quad \text{where } j \text{ maximises } H(p_j) = -p_j \ln p_j - (1-p_j) \ln (1-p_j)$$

### Evaluation Datasets
1. **In-Domain (FaceForensics++ c23):**
   - Official splits: 720 training videos, 140 validation videos, 140 testing videos per manipulation type.
   - Manipulations: Deepfakes (DF), Face2Face (F2F), FaceSwap (FS), NeuralTextures (NT), and pristine YouTube originals.
   - Test set composition: 140 real videos and 560 fake videos (real:fake ratio $r = 0.25$).
   - Identity disjoint: YouTube video IDs across train, val, and test are strictly mutually exclusive (RF1).

2. **Cross-Dataset Generalization (Celeb-DF v2):**
   - Official test split: 518 videos (real and synthetic).
   - Zero-shot evaluation: No Celeb-DF frames or identities are seen during training or validation (RF5).

---

## 2. Pre-Registered Hypotheses

* **H7.1 (In-Domain Detection Performance):**  
  The declared ARES vision backbone (ResNet-50 ImageNet-pretrained) achieves high video-level discriminative power on the official FaceForensics++ c23 test split ($\text{ROC-AUC} \ge 0.9000$). We report exact ROC-AUC, PR-AUC, and operating points with 10,000-resample percentile bootstrap 95% confidence intervals.

* **H7.2 (Cross-Dataset Generalization Gap):**  
  Under zero-shot evaluation on Celeb-DF v2 without fine-tuning, detection performance drops substantially compared to in-domain FF++:
  $$\text{AUC}_{\text{Celeb-DF}} < \text{AUC}_{\text{FF++}} - 0.1500$$
  verifying the standard deepfake domain shift documented in literature.

* **H7.3 (Forensic Auxiliary Feature Enhancement):**  
  Integrating frequency-domain features (azimuthally averaged FFT/DCT power spectrum, Section X-B item 1) and facial landmark temporal geometry (eye aspect ratio jitter, mouth-opening variance, Section X-B item 3) via late fusion improves cross-dataset ROC-AUC on Celeb-DF v2 compared to purely spatial CNN scores:
  $$\Delta\text{AUC}_{\text{fusion}} = \text{AUC}_{\text{CNN+Forensics}} - \text{AUC}_{\text{CNN}} > 0.0200$$

---

## 3. Pre-Registered Model Architectures & Training Protocol

### Architectures
1. **ResNet-50:** The declared ARES visual backbone (Section X, Table XI).
2. **Xception:** The canonical FaceForensics++ benchmark baseline and the operational definition of the manuscript's "CNN baseline".
3. **EfficientNet-B4:** A strong modern reference detector.

### Training Hyperparameters (Frozen)
- Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$)
- Learning rate: $\eta = 2 \times 10^{-4}$ with cosine annealing schedule.
- Loss function: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
- Batch size: Effective batch size 32 (micro-batch 16 with gradient accumulation 2).
- Precision: Mixed precision (bf16/fp16).
- Early stopping: Monitored on **validation video-level ROC-AUC** with patience = 5 epochs.
- Data augmentations (training only): Horizontal flip ($p=0.5$), Color Jitter (brightness 0.2, contrast 0.2), Gaussian blur, JPEG re-compression simulation (quality 40–100), Random Erasing ($p=0.2$). Validation and test splits receive zero augmentation.

---

## 4. Formal Resolution of Audit Issue I-14 (Table XI Inconsistency)

The manuscript claims in Table XI:
- ARES deepfake: $\text{TPR} = 95\%$, $\text{FPR} = 2.5\%$, $F_1 = 0.97$.
- CNN baseline: $\text{TPR} = 89\%$, $\text{FPR} = 8.0\%$, $F_1 = 0.92$.

### Closed-Form Mathematical Proof
Let $\pi = P(Y = 1)$ be the positive (fake) class prior, and $r = \frac{1 - \pi}{\pi} = \frac{N_{\text{real}}}{N_{\text{fake}}}$ be the real-to-fake sample ratio.
1. $\text{TP} = \text{TPR} \cdot \pi N$
2. $\text{FP} = \text{FPR} \cdot (1 - \pi) N = \text{FPR} \cdot r \pi N$
3. $\text{FN} = (1 - \text{TPR}) \cdot \pi N$
4. $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{\text{TPR}}{\text{TPR} + r \cdot \text{FPR}}$
5. $\text{Recall} = \text{TPR}$
6. Substituting into $F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$:
   $$F_1 = \frac{2 \cdot \text{TPR}}{1 + \text{TPR} + r \cdot \text{FPR}}$$

**Theorem:**  
Under balanced classes ($r = 1.0$):
- For ARES ($\text{TPR} = 0.95, \text{FPR} = 0.025$):
  $$F_1 = \frac{2 \times 0.95}{1 + 0.95 + 1 \times 0.025} = \frac{1.90}{1.975} = \mathbf{0.9620} \ne 0.97$$
- For CNN baseline ($\text{TPR} = 0.89, \text{FPR} = 0.08$):
  $$F_1 = \frac{2 \times 0.89}{1 + 0.89 + 1 \times 0.08} = \frac{1.78}{1.97} = \mathbf{0.9036} \ne 0.92$$

Under the official FaceForensics++ c23 test split ($N_{\text{real}} = 140, N_{\text{fake}} = 560 \implies r = 0.25$):
- ARES: $F_1 = \frac{1.90}{1.95625} = \mathbf{0.9712}$
- CNN baseline: $F_1 = \frac{1.78}{1.9100} = \mathbf{0.9319}$

We pre-register that Table XI replacement will report empirical measured metrics accompanied by the exact real:fake ratio ($r$), resolving Audit Issue I-14.

---

## 5. Statistical Protocols & Red-Flag Gates

1. **Bootstrap Confidence Intervals:** All video-level metrics (ROC-AUC, PR-AUC, Accuracy, F1, TPR, FPR) report percentile bootstrap 95% CIs with $B = 10,000$ resamples over videos.
2. **Operating Thresholds:** Thresholds are chosen exclusively on the **validation set** (Youden's $J = \text{TPR} - \text{FPR}$ and a fixed operating point at $\text{FPR} = 0.05$). Never tuned on the test set (RF3).
3. **Red-Flag Gates:**
   - **RF1 (Identity Leakage):** No video ID or source personality may appear in more than one split.
   - **RF2 (Face-Detection Drop Bias):** Frames where face extraction has low confidence must be retained with neutral score ($p = 0.5$) rather than silently discarded.
   - **RF3 (Validation-Only Tuning):** All hyperparameters and thresholds selected on validation.
   - **RF4 (Class Prior Stated):** $F_1$ must always be accompanied by $r$; AUC is the primary threshold-invariant metric.
   - **RF5 (Zero-Shot Cross-Dataset):** No Celeb-DF samples permitted in training/validation.
   - **RF6 (Plausibility Check):** Any cross-dataset AUC exceeding in-domain AUC will be flagged as suspected data contamination.
