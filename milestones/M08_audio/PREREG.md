# Pre-Registration: Milestone M08 — Audio Spoof Detection, Speaker Verification, and AV Alignment

**Milestone:** M08 — Audio  
**Status:** PRE-REGISTERED  
**Date:** 2026-10-13  
**Target Ledger Claims:** `C014`, `C015` (audio row), `C088`  
**Git Branch:** `main`  
**Auditor Session:** independent  

---

## 1. Context & Objectives

Milestone M08 builds, evaluates, and audits the audio and audio-visual forensics components of ARES (Section X-C, Appendix A-c, Table XI, Claims C014, C015, C088).
The objectives are:
1. **Mathematical Audit of Table XI Audio Rows (Audit Issue I-14):**
   Derive the closed-form theorem relating True Positive Rate (TPR), False Positive Rate (FPR), real-to-fake class ratio $r = N_{\text{bonafide}} / N_{\text{spoof}}$, and $F_1$:
   $$F_1 = \frac{2 \cdot \text{TPR}}{1 + \text{TPR} + r \cdot \text{FPR}}$$
   Formally prove that Table XI's reported $F_1 = 0.98$ (for "ARES Voice Embedding") and $F_1 = 0.88$ (for "Traditional Audio ML") are mathematically impossible under balanced classes ($r=1$), implying $0.9700$ and $0.8513$ respectively. Prove that the reported numbers only match when conditioned on an undisclosed imbalanced test ratio ($r \approx 0.32$ and $r \approx 0.47$).
2. **Task Disambiguation (C014, C088):**
   Resolve the ambiguity in the manuscript between:
   - **Task A (Audio Spoof Detection):** Classifying whether an audio utterance is authentic human speech (bonafide) or synthetic/tampered (TTS / voice conversion), evaluated on the standard ASVspoof benchmark.
   - **Task B (Speaker Identity Verification, Eq. 10):** Computing cosine similarity between speaker embeddings $\text{sim}(v_o, v_s) = \frac{v_o \cdot v_s}{\|v_o\| \|v_s\|}$ to detect speaker impersonation, evaluated on the official VoxCeleb1-O test protocol.
   - Clarify Claim C088: VoxCeleb is purely a speaker identification/verification dataset and does not contain synthetic voice manipulation; ASVspoof is the correct benchmark for spoof detection.
3. **Silence-Shortcut Control (Red Flag RF1):**
   Müller et al. (2021) showed that models on ASVspoof can achieve low EER simply by learning leading/trailing silence duration differences between generators and bonafide data. We pre-register a dedicated silence-only baseline classifier and evaluate models on both raw and silence-trimmed audio.
4. **Declared Model Architecture Implementation (Appendix A-c):**
   Implement the exact 6-block CNN on 80-channel log-mel spectrograms producing a 512-dimensional embedding (~8.9M parameters), alongside reference LCNN and traditional acoustic ML baselines.

---

## 2. Pre-Registered Hypotheses

- **H8.1 (Audit Issue I-14 Resolution):**
  Table XI reported values ($F_1 = 0.98$ for voice embedding, $F_1 = 0.88$ for traditional audio ML) are mathematically inconsistent with reported TPR and FPR under balanced classes ($r=1.0$), with discrepancies $+0.0100$ and $+0.0287$. They require class imbalance ratios of $r = 0.3197$ and $r = 0.4697$ respectively.
- **H8.2 (Silence Shortcut Quantification, RF1):**
  A linear classifier trained exclusively on silence duration features (`[leading_silence, trailing_silence, total_silence, total_duration, speech_duration]`) will achieve a non-trivial EER on raw ASVspoof audio, demonstrating the necessity of silence-trimmed evaluation.
- **H8.3 (Declared CNN Superiority):**
  The declared 6-block CNN on 80-channel log-mel spectrograms (Appendix A-c) achieves significantly lower EER and higher ROC-AUC on audio spoof detection than the traditional acoustic cepstral ML baseline (Table XI row 4).
- **H8.4 (Speaker Verification Pipeline Sanity Check, RF4):**
  Evaluating pretrained ECAPA-TDNN (SpeechBrain) on the VoxCeleb1-O test protocol achieves an EER close to published benchmarks ($\le 1.5\%$), establishing the validity of the speaker embedding pipeline.

---

## 3. Datasets & Protocols

| Dataset | Track / Split | Target Use | Split Size / Trials | License |
| :--- | :--- | :--- | :--- | :--- |
| **ASVspoof 2019 LA** | Validation + Test splits | Task A: Spoof Detection (A01–A19 attacks) | 16 kHz genuine audio waveforms | Research Non-Commercial (Interspeech / ASVspoof Consortium) |
| **VoxCeleb1-O** | Official test trial list (`veri_test2.txt`) | Task B: Speaker Verification (Sanity Check RF4) | 37,611 trial pairs | Creative Commons Attribution 4.0 |
| **AV Lip-Sync Protocol** | Synthetic and recorded video tracks | Task C: AV temporal sync deviation $\Delta_{sync}$ (Eq. 9) | 16 kHz audio + 25 FPS video tracks | Internal Benchmark Protocol |

---

## 4. Models to Evaluate

1. **ARES Declared Audio CNN (`DeclaredAudioCNN`):**
   - Input: 80-channel log-mel spectrograms (25 ms window, 10 ms hop).
   - 6 convolutional blocks with BatchNorm2d, ReLU, and pooling.
   - Adaptive pooling to (5, 2) $\to$ 512-dim embedding projection $\to$ binary classification head.
   - Total parameter count: $\approx 8.9\text{M}$ (conforming to Appendix A-c: $\approx 8\text{M}$).
2. **LFCC-LCNN Baseline (`LFCCLCNN`):**
   - Canonical ASVspoof baseline using 40-channel LFCC features and Max-Feature-Map (MFM) activations.
3. **Traditional Audio ML Baseline (`TraditionalAudioML`):**
   - Acoustic cepstral feature statistics (mean, std, delta-mean, delta-std across frames) classified by HistGradientBoosting. Represents Table XI row 4 ("Traditional Audio ML").
4. **Silence-Only Shortcut Classifier (`SilenceShortcutAuditor`):**
   - Logistic regression on 5-dimensional silence duration metrics (RF1).
5. **ECAPA-TDNN Speaker Verifier (`ECAPASpeakerVerifier`):**
   - SpeechBrain pretrained on VoxCeleb2 (192-d embeddings, cosine similarity scoring).

---

## 5. Metrics & Threshold Selection

- **Primary Spoof Metric:** Equal Error Rate (EER) and ROC-AUC.
- **Threshold Policy (RF2):** Decision thresholds are calibrated strictly on validation/development trials (EER operating point or Youden $J$) and frozen before test evaluation. Zero test-set threshold searching.
- **Table XI Replacement Metrics:**
  - True Positive Rate (TPR), False Positive Rate (FPR), Balanced $F_1$ ($r=1.0$), and Stated-ratio $F_1$ ($r$ explicitly documented).
- **Statistical Significance:**
  - 10,000-resample percentile bootstrap 95% confidence intervals over utterances.

---

## 6. Milestone-Specific Red-Flag Gates

- **RF1: Silence Shortcut Control:** Silence durations measured; silence-only baseline reported; models evaluated on both raw and silence-trimmed audio.
- **RF2: Validation-Only Thresholds:** Operating thresholds frozen from dev/validation data only.
- **RF3: Official Protocols:** Canonical EER calculation used; test set accesses logged in `milestones/test_set_access_log.csv`.
- **RF4: Speaker Verification Sanity:** ECAPA-TDNN EER on VoxCeleb1-O verified against literature.
- **RF5: Disjoint Data Integrity:** VoxCeleb is not treated as voice spoofing data (Claim C088).
- **RF6: Plausibility vs Literature:** ASVspoof EERs compared against canonical challenge results.

---

## 7. Execution Environment

- **Hardware:** HP OMEN Laptop (NVIDIA GeForce RTX 4060 Laptop GPU 8 GB, Intel Core i7-13620H, 16 GB RAM).
- **Precision:** fp32 on GPU and CPU.
- **RNG Seeds:** Master seed 42.
