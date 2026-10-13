# M08 — Audio: Spoof Detection, Speaker-Embedding Similarity, (AV Sync)

**Objective:** Build the audio module of Section X-C / Appendix A. Use the **right datasets**: ASVspoof for voice tampering/spoofing, VoxCeleb for speaker embeddings, FakeAVCeleb for audio-visual. Replace Table XI's audio rows and resolve the ambiguity in "ARES (Voice Embedding)".
**Ledger IDs:** C014, C015 (traditional audio ML row), C088.
**Depends on:** M02.
**Compute:** RTX 4060 Laptop, ≈ 45 GPU-h. AASIST and LCNN fit in 8 GB at 4 s crops. ECAPA is inference-only (pretrained on VoxCeleb2, so no VoxCeleb training data is needed). GMM baselines run on the CPU.
**Output dir:** `milestones/M08_audio/`

---

## Pre-registration
- **Task A (spoof detection, ASVspoof 2019 LA):** primary metric EER on the official **eval** set (attacks A07–A19, mostly unseen in training); secondary min t-DCF (official script), per-attack EER, and accuracy/F1/TPR/FPR at the **dev**-chosen EER threshold (for comparability with Tab. XI). Cross-dataset: ASVspoof 2021 DF eval, In-the-Wild.
- **Task B (speaker similarity, Eq. 10):** ECAPA-TDNN (SpeechBrain, pretrained on VoxCeleb2) on the VoxCeleb1-O trial list: EER. Then the paper's tampering use: on ASVspoof 2019 LA's **ASV protocol** (target / non-target / spoof trials), measure how well sim(v_o, v_s) separates target from spoof trials (spoof-EER of the ASV system).
- **Task C (optional, access):** audio-visual on FakeAVCeleb (feeds T6 in M11) and lip-sync Δsync (Eq. 9) with SyncNet.
- **Models for Task A:** (1) **declared** 6-block CNN on 80-channel log-mel, 512-d embedding (App. A-c), implemented exactly; (2) LFCC-LCNN (official-style baseline); (3) AASIST (official code; strong reference); (4) "Traditional audio ML" = GMM on LFCC/CQCC (official baselines), the meaning of Tab. XI's last row.
- **Seeds:** K = 10 for the declared CNN; K = 5 for LCNN/AASIST if budget-limited.

## Steps

### M08.1 — Feature pipeline
- **Action:** `src/ares/audio/features.py`: 16 kHz resample, 80-ch log-mel (25 ms / 10 ms), LFCC, CQCC (for GMM). Fixed-length crops/pads (4 s default, configurable). Deterministic.

### M08.2 — Silence-shortcut control (critical)
- **Action:** It's documented in the literature that leading/trailing silence duration correlates with the label in ASVspoof 2019 LA (Müller et al., 2021). Using M02's silence durations: (a) train a trivial classifier on silence duration alone and report its EER; (b) train/evaluate the declared CNN on silence-trimmed audio as well as raw. **Report both.** The trimmed result is the honest headline if the gap is large.
- **Done when:** `tables/silence_shortcut.csv`.

### M08.3 — Train spoof detectors
- **Action:** `src/ares/audio/spoof.py` + configs. Train on LA train, select on LA dev, evaluate once on LA eval. Then run cross-dataset (2021 LA eval, In-the-Wild) without fine-tuning.
- **Tier 2 (cross-dataset, frozen models):** the seeded subsets of ASVspoof 2021 DF eval and WaveFake (stratified by vocoder/attack), plus Fake-or-Real. Report EER with CIs and per-vocoder EER.

### M08.4 — Speaker embeddings (Task B)
- **Action:** `src/ares/audio/speaker.py`: ECAPA-TDNN embeddings; cosine scoring; VoxCeleb1-O EER (compare with the model card's published EER, as a pipeline sanity check); ASVspoof ASV-protocol target vs spoof separation.

### M08.5 — Lip-sync / AV (Task C, if FakeAVCeleb access)
- **Action:** SyncNet offset/confidence per video → Δsync feature; audio-only, video-only (M07 model), and AV fusion classifiers on FakeAVCeleb with the dataset's recommended splits. If access isn't granted → ledger REFRAMED for lip-sync claims.

### M08.6 — Evaluation & Table XI audio rows (C014, C015)
- **Action:** `tables/audio_main.csv/.tex` (EER, min t-DCF, per-attack EER, cross-dataset EER) + `tables/table11_audio_replacement.csv` (TPR/FPR/F1 at the dev threshold, with class ratio). State clearly what "ARES (Voice Embedding)" now means: either the spoof detector, or the speaker-similarity tamper check. They're different tasks.
- **Stats:** bootstrap CIs over utterances (EER CI via bootstrap); paired comparisons via bootstrap on shared eval trials.

### M08.7 — Cache outputs
- **Action:** Per-utterance scores (and per-video AV scores) for fusion and latency (M13).

---

## Audit gate — milestone-specific red flags
- RF1: **Silence shortcut** quantified, and the trimmed-audio result reported.
- RF2: Thresholds from dev only; eval accessed once per final model (logged).
- RF3: Official protocols/scripts used for EER and t-DCF (version recorded).
- RF4: VoxCeleb1-O EER close to the pretrained model's published figure (pipeline sanity check).
- RF5: No VoxCeleb data used as "manipulated voice" data (C088).
- RF6: Plausibility vs published ASVspoof results (AASIST reports < 1 % EER on LA eval; a declared-CNN EER far better than AASIST is a red flag).

## Exit criteria
- [ ] Spoof detectors (declared CNN + references) evaluated with the silence control; cross-dataset done.
- [ ] Speaker-embedding tasks done.
- [ ] AV/lip-sync done, or ledger REFRAMED with reason.
- [ ] Table XI audio rows replaced; ledger C014, C015, C088 updated.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m08-complete`.
