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
* **Committed:** commit `f72193d`.

---

## Phase 2: Build (Registry, Canonical Schema & Ingestion Loaders)
* Created `ares_v2/configs/data/registry.yaml`: master dataset registry for Tier-1 and Tier-2 benchmarks.
* Created `ares_v2/configs/data/label_maps.yaml`: pre-registered label mappings and collapses.
* Created `milestones/M02_data/tables/storage_projection.csv`: storage allocation model proving peak stays $\le 145$ GB.
* Created `ares_v2/src/ares/data/schema.py`: canonical schema validator with Pydantic and PyArrow Parquet I/O.
* Built dataset ingestors:
  * `ares_v2/src/ares/data/liar.py`: Ingested LIAR (10,269 train / 1,284 val / 1,283 test).
  * `ares_v2/src/ares/data/constraint.py`: Ingested Constraint-COVID (6,420 train / 2,140 val / 2,140 test).
  * `ares_v2/src/ares/data/coaid.py`: Ingested CoAID with grouped 70/10/20 domain split (5,711 items).
  * `ares_v2/src/ares/data/xfact.py`: Ingested X-FACT multilingual dataset across 14 languages (37,672 items).
  * `ares_v2/src/ares/data/paws.py`: Ingested PAWS labeled_final (49,401 train / 8,000 val / 8,000 test).
  * `ares_v2/src/ares/data/stsb.py`: Ingested GLUE STS-B (5,749 train / 1,500 val / 1,379 test).
  * `ares_v2/src/ares/data/fever.py`: Ingested FEVER 1.0 (145,449 train / 9,999 val / 9,999 test).
* Generated SHA-256 locked split manifests in `ares_v2/data_manifests/*.csv` and `manifests.json`.

---

## Phase 3: Test
* Ran unit tests:
  * `pytest ares_v2/tests/data/test_schema.py` passed 4/4 tests.
  * `pytest ares_v2/tests/` passed 22/22 tests.

---

## Phase 4: Run (Data Leakage & Contamination Audit)
* Built and executed `ares_v2/scripts/data/leakage_audit.py`:
  * Evaluated 306,395 items across 7 datasets.
  * Discovered inherent benchmark duplicates: Constraint (41 exact train-test), X-FACT (26), LIAR (5), CoAID (5), FEVER (2), PAWS (0).
  * Audited cross-dataset contamination: 77.18% of LIAR exists in X-FACT (9,882 claims) due to shared PolitiFact sourcing.
  * Generated `milestones/M02_data/tables/split_sizes.csv`, `milestones/M02_data/tables/leakage_summary.csv`, and `milestones/M02_data/results/leakage_report.json`.

---

## Phase 5: Report & Ledger Delta
* Created `milestones/M02_data/ledger_delta.csv`.
* Applied delta via `python ares_v2/scripts/ledger.py apply-delta M02_data`:
  * Updated C032, C088, C089 to `REVISED`.
* Verified ledger integrity: `validate` and `check-evidence` passed with 11/11 valid claims.
* Generated `milestones/M02_data/README.md`.

---

## Phase 6: Adversarial Audit
* Auditor review conducted and documented in `milestones/M02_data/AUDIT.md`.
* Verdict: CLOSE.

---

## Phase 7--9: Close
* Tagged `m02-complete`.
* Pushed commits and tags to GitHub origin.

