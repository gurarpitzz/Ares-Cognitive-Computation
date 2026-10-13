# Milestone M02 — Worklog (Data Acquisition, Curation, Splits & Leakage Audit)

## Phase 0: Orientation & Audit Review
* **Date:** 2026-10-13
* **Audited Manuscript Sections:**
  * Section V / Table I: Reported dataset volumes (Claim C032: "15 datasets, 2.1M claims, 450k images, 12k videos, 85k audio samples").
  * Section X: Audio verification and VoxCeleb trial misuse (Claim C088).
  * Section XII: Multilingual verification across 10 languages (Claim C089).
* **Ledger Target Claims:** C032, C088, C089.
* **Storage Budget:** Drive D: 202.2 GB free. Hard peak budget: <= 150 GB. Raw retention policy: keep text/manifests; for heavy video/audio, derive features/crops then delete raw if required.
* **Dependencies:** M00 foundation packages, PyArrow, Parquet, streaming JSON (`ijson`).

---

## Phase 1: Pre-Registration
* **File:** `milestones/M02_data/PREREG.md`
* **Status:** PRE-REGISTERED
