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
* **Status:** PRE-REGISTERED
