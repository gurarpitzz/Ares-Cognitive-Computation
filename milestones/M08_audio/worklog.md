# Engineering Worklog — Milestone M08: Audio Deepfake Detection & Speaker Verification

**Engineer:** Antigravity  
**Session:** autonomous pair programming  
**Date:** 2026-10-13  
**Corpus / Repo:** `gurarpitzz/Ares-Cognitive-Computation`  
**Milestone:** `M08_audio`  
**Target Ledger Claims:** `C014`, `C015` (audio row), `C088`  

---

## Phase 1: Planning, Scoping & Mathematical Derivation
- Analyzed manuscript Section X-C, Appendix A-c, and Table XI audio claims:
  - Row 2: "ARES (Voice Embedding)" reported TPR=97%, FPR=3.0%, F1=0.98.
  - Row 4: "Traditional Audio ML" reported TPR=83%, FPR=12.0%, F1=0.88.
- Resolved Audit Issue I-14 for audio modalities:
  - Derived closed-form relationship $F_1 = \frac{2 \cdot \text{TPR}}{1 + \text{TPR} + r \cdot \text{FPR}}$.
  - Proved that under balanced classes ($r=1.0$), reported TPR=97% and FPR=3.0% imply $F_1 = 0.9700$, not $0.98$ (discrepancy $+0.0100$). Exact $0.98$ requires $r = 0.3197$ (~75.8% spoof prevalence).
  - Proved that under balanced classes ($r=1.0$), reported TPR=83% and FPR=12.0% imply $F_1 = 0.8513$, not $0.88$ (discrepancy $+0.0287$). Exact $0.88$ requires $r = 0.4697$ (~68.0% spoof prevalence).
- Disambiguated Task A (voice tampering / spoof detection on ASVspoof) and Task B (speaker verification on VoxCeleb), clarifying Claim C088.

## Phase 2: Architecture & Feature Implementation
- Implemented `ares_v2/src/ares/audio/audit_table11.py` with closed-form solvers.
- Implemented `ares_v2/src/ares/audio/features.py`:
  - 16 kHz audio loading and resampling.
  - 80-channel log-mel spectrogram extraction per Appendix A-c (25 ms window, 10 ms hop).
  - 40-channel Linear Frequency Cepstral Coefficients (LFCC) extraction.
  - Energy-based voice activity detection (VAD), silence duration measurement (`[leading, trailing, total, duration, speech]`), and silence trimming.
- Implemented `ares_v2/src/ares/audio/models.py`:
  - `DeclaredAudioCNN`: 6-block CNN on 80-channel log-mel spectrograms, 512-dim embedding projection, 8,895,426 parameters (matching Appendix A-c: $\approx 8\text{M}$).
  - `LFCCLCNN`: Light-CNN with Max-Feature-Map (MFM) activations on LFCC features.
  - `TraditionalAudioML`: Acoustic summary statistics (mean, std, delta-mean, delta-std) with HistGradientBoosting.
- Implemented `ares_v2/src/ares/audio/speaker.py`:
  - Pretrained ECAPA-TDNN (SpeechBrain) on VoxCeleb2.
  - Cosine similarity scoring per Eq. 10.
  - Canonical EER calculation with threshold search.
- Implemented `ares_v2/src/ares/audio/silence.py`:
  - Silence-only shortcut classifier (RF1 control).
- Implemented `ares_v2/src/ares/audio/av_sync.py`:
  - Full-wave rectified envelope extraction and normalized cross-correlation lag analysis ($\Delta_{sync}$ Eq. 9).

## Phase 3: Unit Testing & Environment Setup
- Installed `torchaudio==2.6.0+cu124`, `soundfile==0.14.0`, and `speechbrain==1.1.1` in Python 3.11 environment on Windows.
- Created `ares_v2/tests/test_audio.py` covering mathematical proof of I-14, feature extraction, model forward passes, parameter counts, speaker verification, silence shortcut, and AV sync.
- All 6 audio unit tests pass cleanly. Total repository test suite: 49/49 passing.

## Phase 4: Pre-Registration
- Authored and committed `milestones/M08_audio/PREREG.md` defining hypotheses H8.1–H8.4, datasets, models, metrics, and red-flag gates RF1–RF6.

## Phase 5: Empirical Experiment Execution on Genuine Speech
- Streamed and decoded 600 genuine audio waveforms (300 bonafide human recordings from multiple speakers, 300 spoofed utterances across speech synthesis and voice conversion systems) from the official ASVspoof 2019 LA benchmark. Cached locally at `data/processed/asvspoof_sample_cache.pt`.
- Executed Silence Shortcut Audit (RF1):
  - Proved that silence duration alone achieves 100% AUC (0% EER) on raw ASVspoof 2019 LA audio due to synthetic padding differences (validating Müller et al., 2021).
- Evaluated Audio Spoof Detectors:
  - DeclaredAudioCNN (Raw): ROC-AUC = 1.0000, TPR = 100.0%, FPR = 2.78%, Balanced $F_1 = 0.9863$, Stated-ratio $F_1 = 0.9956$.
  - DeclaredAudioCNN (Silence-Trimmed Honest Headline): ROC-AUC = 0.8178 $[0.7698, 0.8613]$, PR-AUC = 0.8420, EER = 23.13%, Balanced $F_1 = 0.6792$.
  - LFCC-LCNN Baseline: ROC-AUC = 1.0000.
  - Traditional Audio ML: ROC-AUC = 1.0000.
- Evaluated Task B (Speaker Verification, Eq. 10):
  - ECAPA-TDNN evaluated on genuine speech pairs: mean target similarity 0.5364, mean non-target similarity 0.0315, EER = 0.00%. Confirmed RF4 sanity check.
- Evaluated Task C (AV Lip-Sync Offset, Eq. 9):
  - In-sync mean lag = 0.0 ms, tampered (+160 ms lag) mean lag = 160.0 ms. 100% tampering detection.

## Phase 6: Output Artifact Generation & Ledger Update
- Generated 11 benchmark tables in `tables/` (CSV and LaTeX format).
- Generated vector PDF and PNG figures in `figures/`.
- Saved schema-compliant results JSONs (`M08-audio-declared-cnn-s42.json`, `M08-audio-traditional-ml-s42.json`).
- Logged official test accesses in `milestones/test_set_access_log.csv`.
- Applied ledger delta updating Claims `C014`, `C015`, and `C088` to `REVISED`.
- Validated claims ledger: 99 claims valid, 34/34 verified claims passing with valid evidence.

## Phase 7: Max-Rigor Adversarial Audit & Stratified Re-Evaluation
- Conducted internal adversarial self-audit of initial M08 run and identified 4 rigor deficiencies:
  1. Sampling bias: Sequential streaming had yielded spoof samples from only 1 attack (A01) and 1 speaker (LA_0069).
  2. Silence control incompleteness: Had only tested raw model on trimmed audio, without training directly on trimmed audio.
  3. Small trial count in speaker verification (18 pairs).
  4. Coarse temporal lag shifts in AV sync.
- Remedied all 4 deficiencies:
  1. Built `fetch_stratified_asvspoof.py` using Parquet row-group targeting to stream a 100% balanced stratified sample of 600 genuine waveforms across 5 distinct attack algorithms (A01, A03, A04, A05, A06) and 8+ distinct speakers (`data/processed/asvspoof_stratified_cache.pt`).
  2. Implemented dual training regime: trained `DeclaredAudioCNN`, `LFCCLCNN`, and `TraditionalAudioML` under both raw and silence-trimmed audio. Proved that when denied the silence shortcut, `DeclaredAudioCNN` trained on silence-trimmed audio achieves 0.9990 ROC-AUC [0.9971, 1.0000] and 1.33% EER, demonstrating genuine acoustic synthesis artifact detection. Replicated Müller et al. (2021) by showing cross-condition degradation (AUC drops to 0.7911 [0.7357, 0.8408], EER jumps to 29.33%).
  3. Scaled speaker verification (Task B) to 400 genuine speech trial pairs (200 target, 200 non-target): achieved EER = 0.00%, ROC-AUC = 1.0000, target similarity 0.6137 +/- 0.11 vs non-target 0.0583 +/- 0.07, separation margin Delta = 0.5554.
  4. Evaluated AV lip-sync across fine-grained lag sweep (0, 40, 80, 120, 160, 200 ms) with threshold tau = 80 ms.
  5. Added calibration curves (reliability diagrams) in `figures/audio_calibration_curves.png` and results JSON `M08-audio-speaker-ecapa-s42.json`.
  6. Updated `plan/claims_ledger.csv`, `milestones/test_set_access_log.csv`, and `milestones/compute_ledger.csv`.
  7. Re-verified full test suite: 49/49 passing.
