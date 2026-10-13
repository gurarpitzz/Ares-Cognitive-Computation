# Milestone M04 — Worklog (Evidence Engine R·S·N·W + Logic Layer)

## Phase 0: Orientation & Audit Review
* **Date:** 2026-10-13
* **Audited Manuscript Sections:**
  * Section VII & Equation 10: Evidence formula $\text{Truth}'(C) = \frac{\sum S \cdot N \cdot W}{\sum S}$.
  * Section XII & Table XV: 4-layer verification metrics (text filtering, claim extraction, contextual reasoning, cross-source verification).
  * Section XIV / Showcase trace: J&K claim evidence trace.
  * Section XXI: Legacy confidence formula $\text{Conf}_{\text{POC}} = |\text{Truth}'| \cdot \log(1+M)$ to be audited and replaced with calibrated posterior probabilities.
* **Target Ledger Claims:** C018, C040, C067, C078.
* **Integrity Gates:** RF1 (No FEVER-NLI contamination in headline evaluation), RF2 (Fact-check domain exclusion), RF3 (Temporal leakage filtering), RF4 (Thresholds chosen strictly on validation split).

---

## Phase 1: Pre-Registration
* **File:** `milestones/M04_evidence_engine/PREREG.md`
* **Status:** PRE-REGISTERED (Committed prior to evaluation)

---

## Phase 2: Build
* Implemented modular operators in `ares_v2/src/ares/evidence/`:
  * `retriever.py`: BM25, Dense (SentenceTransformers + FAISS), and Hybrid Reciprocal Rank Fusion.
  * `ranker.py`: BiEncoder and CrossEncoder semantic relevance scoring.
  * `nli.py`: NLI stance classifier supporting Soft-N ($P(\text{ent}) - P(\text{contra})$) and Hard-N.
  * `credibility.py`: Source credibility scorer with Lin et al. ratings and fact-check domain exclusion filter.
  * `functional.py`: Evidence functional $\text{Truth}'(C) = \frac{\sum S \cdot N \cdot W}{\sum S}$, legacy confidence analysis, and calibrated sigmoid posterior.
* Implemented symbolic reasoning modules in `ares_v2/src/ares/logic/`:
  * `quantities.py`: Numerical quantity, percentage, and date extractor.
  * `logic_score.py`: Contradiction and consistency adjustment layer.

---

## Phase 3: Test & Smoke Verification
* Built unit test suite `ares_v2/tests/test_evidence_logic.py`.
* Verified all 4 tests pass covering BM25 retrieval, credibility scoring, evidence functional math, and numerical conflict extraction.

---

## Phase 4: Run
* Executed master evaluation script `ares_v2/scripts/run_m04_evaluation.py`.
* Benchmarked retrieval operators on FEVER validation claims (BM25, Dense, Hybrid RRF).
* Evaluated NLI stance classification and functional behavior (Soft-N vs Hard-N).
* Evaluated credibility weighting impact under noisy distractor injection.
* Replaced Table XV with empirically grounded 4-layer metrics.
* Re-ran the J&K showcase claim with frozen offline evidence and full attribution.

---

## Phase 5: Report
* Generated 4 summary tables in `milestones/M04_evidence_engine/tables/`:
  * `retriever_comparison.csv`
  * `nli_functional_comparison.csv`
  * `credibility_comparison.csv`
  * `table15_replacement.csv`
* Generated showcase trace in `milestones/M04_evidence_engine/results/showcase_jk_trace.json`.
* Logged compute entry in `milestones/compute_ledger.csv`.

---

## Phase 6 & 7: Audit & Fix
* Reconciled claims ledger for C018, C040, C067, C078 via `ledger_delta.csv`.
* Applied delta and verified 21/21 claims pass evidence checks via `ledger.py check-evidence`.
* Documented findings and verified all integrity gates in `milestones/M04_evidence_engine/AUDIT.md`.

---

## Phase 8 & 9: Re-Audit & Close
* Tagged `m04-complete`.
* Pushed commit and tags to `origin/main`.
* Milestone M04 officially closed.

