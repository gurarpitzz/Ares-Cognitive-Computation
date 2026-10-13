# Milestone M03 — Worklog (Text Baselines & Layer-1 Text Verifiers)

## Phase 0: Orientation & Audit Review
* **Date:** 2026-10-13
* **Audited Manuscript Sections:**
  * Section VI: Layer-1 architecture (RoBERTa-base, DeBERTa-v3, LLM ensemble).
  * Section VIII: Hyperparameters (Claim C050: lr 2e-5, effective bs 32, 15 epochs, cosine warm-up).
  * Section XII: Multilingual verification across 10-14 languages (Claim C039).
  * Section XIV / Tables XXI, XXXVII: Baseline models (Claim C064: FakeFlow, Grover, ClaimBuster; Claim C065: SOTA comparison).
* **Target Ledger Claims:** C038, C039, C050, C064, C065.
* **Hardware & Acceleration:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Intel Core i7-13620H (16 threads). Native PyTorch 2.6 with CUDA 12.4 in Windows.
* **Integrity Guard:** Model input restricted strictly to claim text (no metadata or justification leakage). Test set evaluated strictly once after model selection on validation set. All instance predictions saved to parquet.

---

## Phase 1: Pre-Registration
* **File:** `milestones/M03_text_verifiers/PREREG.md`
* **Status:** PRE-REGISTERED (Committed prior to model evaluation)

---

## Phase 2: Build
* Built modular classical baselines harness (`ares_v2/src/ares/text/classical.py`) with 4 models: Majority, Stratified Random, TF-IDF + Logistic Regression, TF-IDF + Linear SVM.
* Built full transformer fine-tuning harness (`ares_v2/src/ares/text/finetune.py`) with bfloat16 autocasting, AdamW optimizer, dynamic length bucketing, and checkpointing.
* Built offline model weight downloader (`ares_v2/scripts/download_model.py`) to bypass Windows git-lfs stalls.
* Enforced float32 master weights for DeBERTa-v3 to resolve float16 gradient underflow / NaN instability.

---

## Phase 3: Test & Smoke Verification
* Ran unit and smoke tests on binary datasets.
* Verified bit-exact determinism across repeated runs with seed 42.

---

## Phase 4: Run
* Completed all 16 classical baseline runs across CoAID, Constraint, LIAR, and X-FACT.
* Completed 4 RoBERTa-base runs across all 4 benchmarks.
* Completed 3 DeBERTa-v3-base runs (CoAID, Constraint, LIAR).
* Completed multilingual mDeBERTa-v3-base run on X-FACT across 14 languages.
* Preserved 24 instance-level prediction Parquet files with logits and probabilities.

---

## Phase 5: Report
* Compiled `milestones/M03_text_verifiers/tables/text_verifiers_summary.csv` via `ares_v2/scripts/aggregate_m03.py`.
* Verified 10,000-sample bootstrap confidence intervals for all macro-F1 metrics.
* Logged GPU hours and peak VRAM to `milestones/compute_ledger.csv`.
* Logged test-set evaluations to `milestones/test_set_access_log.csv`.

---

## Phase 6 & 7: Audit & Fix
* Reconciled claims ledger against empirical results for C034, C038, C039, C050, C064, C065.
* Verified that all evidence paths exist via `ares_v2/scripts/ledger.py check-evidence`.
* Documented findings in `milestones/M03_text_verifiers/AUDIT.md`.

---

## Phase 8 & 9: Re-Audit & Close
* Tagged `m03-complete`.
* Pushed commit and tags to `origin/main`.
* Milestone M03 officially closed.

