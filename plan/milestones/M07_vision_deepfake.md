# M07 — Visual Deepfake Detection

**Objective:** Build and evaluate the vision module of Section X / Appendix A (ResNet-50, frames at 1–2 FPS, mean pooling), plus standard reference detectors. Evaluate in-domain (FF++), cross-dataset (Celeb-DF v2), and under compression. Replace Table XI's vision rows with measured values.
**Ledger IDs:** C013, C015 (CNN baseline row).
**Depends on:** M02 (FF++ / Celeb-DF access + face crops).
**Compute:** RTX 4060 Laptop, ≈ 60 GPU-h (3 architectures × seeds, on pre-extracted face crops). Use bf16, micro-batch from `vram_fit.csv`, accumulation to the declared batch, resumable training, and a crop-only data pipeline (never decode video during training).
**Output dir:** `milestones/M07_vision_deepfake/`

---

## Pre-registration
- **Primary metric:** video-level ROC-AUC on the FF++ c23 official test split (all manipulations pooled; real:fake ratio stated). Secondary: per-manipulation AUC; accuracy and F1 at a validation-chosen threshold; TPR at FPR = 1 % and 5 %; cross-dataset AUC on Celeb-DF v2 test; AUC on FF++ c40.
- **Models:** (1) ResNet-50 ImageNet-pretrained, **declared** ARES encoder; (2) Xception, the FF++ reference baseline, and the definition of "CNN baseline" in Tab. XI; (3) EfficientNet-B4 (strong reference).
- **Seeds:** K = 10 for ResNet-50; K = 5 for Xception and EfficientNet-B4 if budget-limited (stated).
- **H7.1:** ResNet-50 video-level AUC on FF++ c23 (descriptive, with CI).
- **H7.2:** Cross-dataset AUC (Celeb-DF) is lower than in-domain (expected; quantify the gap).
- **H7.3:** Adding frequency-domain + landmark-geometry features (Sec. X-B items 1 and 3) improves cross-dataset AUC (exploratory).

## Steps

### M07.1 — Data loaders
- **Action:** `src/ares/vision/data.py`: frame-level dataset over the M02 face crops; video-level grouping; class-balanced sampling (FF++ has ~4–5 fake videos per real one); augmentations: horizontal flip, colour jitter, Gaussian blur, JPEG re-compression (quality 30–100), random erasing. Val/test use no augmentation.
- **Done when:** one batch visualised (`figures/batch_sample.png`); per-split video counts match the official JSON.

### M07.2 — Training harness
- **Action:** `src/ares/vision/train.py` (timm backbones): BCE loss, AdamW, cosine schedule, early stopping on **validation video-level AUC**, mixed precision, deterministic seeding. Configs per architecture in `configs/m07/`.
- **Done when:** a smoke run (1 epoch, subset) completes; two identical-seed runs match within tolerance.

### M07.3 — Train the grid
- **Action:** Train ResNet-50 / Xception / EfficientNet-B4 × seeds on FF++ c23 (all manipulations), queued overnight. Record GPU-hours, peak VRAM and throttling in the compute ledger.
  - **Data-loading tactic:** crops are stored as JPEG on the ext4 VHDX, decoded with `num_workers` 6 and pinned memory. If the GPU is starved (utilisation < 70 %), switch to DALI or pre-resized uint8 shards. Record which loader was used.

### M07.4 — Video-level aggregation & thresholds
- **Action:** Video score = mean of frame probabilities over frames sampled at 1–2 FPS (declared). Variant: mean of top-k entropy frames (App. D mention). Threshold for accuracy/F1/TPR/FPR chosen on **validation** (Youden J, and a fixed-FPR = 5 % operating point).

### M07.5 — Forensic auxiliary features (Sec. X-B items 1 and 3)
- **Action:** `vision/forensics.py`: (a) azimuthally averaged FFT/DCT spectrum features per face crop; (b) facial-landmark geometry (68-point or MediaPipe): eye aspect ratio, mouth-opening temporal variance, landmark jitter across frames. A small classifier (logistic/GBM) standalone + late fusion with CNN scores. Evaluate H7.3.

### M07.6 — Evaluation
- **Action:** FF++ c23 test (pooled + per manipulation), FF++ c40, Celeb-DF v2 test (cross-dataset, no fine-tuning). Bootstrap CIs over **videos**. Paired comparisons between architectures (paired bootstrap on shared test videos) with Holm.
- **Tier 2 (cross-dataset, frozen models, no fine-tuning):** DFDC preview, WildDeepfake, and the documented DeeperForensics-1.0 subset, if acquired. Report AUC with CIs. These are descriptive generalisation results; expect large drops and report them.
- **Output:** `tables/deepfake_main.csv/.tex`, `figures/roc_ffpp.pdf`, `figures/roc_celebdf.pdf`, `tables/per_manipulation.csv`, `tables/deepfake_cross_tier2.csv`.

### M07.7 — Adversarial robustness (Sec. X-E, optional within budget)
- **Action:** PGD (L∞, ε ∈ {1/255, 2/255, 4/255}) at test time on frames; AUC degradation; optional adversarial fine-tuning. Report or mark not done. On 8 GB, attack a seeded subset of 200 test videos (the subset IDs are logged).

### M07.8 — Table XI vision rows (C013, C015)
- **Action:** Produce the replacement rows: model, AUC, TPR, FPR, F1 (with the stated real:fake ratio), cross-dataset AUC. Explain that F1 depends on class ratio (audit I-14).

### M07.9 — Cache outputs
- **Action:** Per-video probabilities for all test/val videos (and out-of-fold train if used in fusion, e.g. for FakeAVCeleb T6). Save frame-timing info for M13.

---

## Audit gate — milestone-specific red flags
- RF1: **Identity leakage:** FF++ source/target identities split per official JSON; no video's frames in two splits.
- RF2: **Face-detection drop bias:** failure rates per class reported; if fakes fail detection more often, those videos are counted (scored as fake with p = 0.5, or reported separately), not silently dropped.
- RF3: Thresholds selected on validation only.
- RF4: F1 reported together with the class ratio; AUC is primary.
- RF5: Cross-dataset evaluation involves no Celeb-DF data in training/tuning.
- RF6: Plausibility: in-domain c23 AUC typically very high, cross-dataset much lower. A cross-dataset AUC close to in-domain is a red flag.

## Exit criteria
- [ ] Three architectures trained with seeds; full evaluation with CIs.
- [ ] Forensic features evaluated (H7.3).
- [ ] Table XI vision rows replaced with measured values.
- [ ] Outputs cached for fusion and latency.
- [ ] Ledger C013, C015 (vision) updated.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m07-complete`.
