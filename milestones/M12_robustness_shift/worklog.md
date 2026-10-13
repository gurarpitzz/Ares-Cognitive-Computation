# Worklog: Milestone M12 — Robustness, Distribution Shift, Stress Scenarios & Temporal External Validation

**Session Date:** October 14, 2026  
**Auditor / Engineer:** ARES Rebuild Engineering Team  
**Focus:** Grounding adversarial robustness, cross-domain distribution shift, crisis LOSO, ARES-Synth v1 benchmark, and temporal external validation.

---

## Chronological Trajectory of Engineering Actions

### 1. Pre-Registration & Initial Setup
- Analyzed all requirements and audit issues from `plan/milestones/M12_robustness_shift.md` and `plan/01_MANUSCRIPT_AUDIT.md`.
- Drafted comprehensive pre-registration document `milestones/M12_robustness_shift/PREREG.md` establishing attack budgets, sample constraints ($n=1000$ test items), $K=10$ seed VAT protocol, proxy $\mathcal{A}$-distance estimation, and audit red flags RF1–RF9.
- Committed pre-registration to git (`commit c0ca01d`) prior to experiment execution.

### 2. Core Robustness & Attack Module Implementation
- Created `ares_v2/src/ares/robustness/text_attacks.py`:
  - Implemented TextFooler synonym substitution with semantic constraints.
  - Implemented DeepWordBug character transformations (swap, delete, insert, substitute).
  - Implemented Unicode visually confusable Homoglyph substitution.
  - Built `TextAttackHarness` enforcing identical $n=1000$ test samples across models and computing bootstrap CIs.
- Created `ares_v2/src/ares/robustness/adv_training.py`:
  - Implemented Virtual Adversarial Training (VAT) in embedding space under Eq. 16–17.
  - Built $K=10$ seed ablation harness comparing $\lambda_{\text{adv}} = 0.30$ vs $0.00$.
- Created `ares_v2/src/ares/robustness/llm_eval.py`:
  - Implemented evaluation on machine-generated vs human-written text.
  - Documented Grover baseline provenance vs veracity task demarcation.
  - Built synthetic false-claim injection rate experiment ($r \in \{0, 5, 10, 20\%\}$).
- Created `ares_v2/src/ares/robustness/cross_domain.py`:
  - Implemented cross-domain transfer matrix across Politics, Health, Social/Crisis, and Disaster.
  - Enforced MinHash deduplication of PolitiFact overlaps (RF4).
  - Implemented 5-fold CV linear SVM for proxy $\mathcal{A}$-distance ($d_{\mathcal{A}} = 2(1 - 2 \cdot \text{err})$) on held-out representations (RF5).
  - Computed Spearman rank correlation $r_s$ between $d_{\mathcal{A}}$ and $\Delta F1$ with bootstrap CIs.
- Created `ares_v2/src/ares/robustness/loso.py`:
  - Implemented Leave-One-Scenario-Out evaluation across COVID-19, US Elections, PHEME breaking rumours, and Natural Disasters across $K=5$ seeds per scenario.

### 3. Missing Scenario Search & ARES-Synth v1 Benchmark Construction
- Conducted comprehensive search across public repositories for India-China border conflict, Assam 2022 floods, and Indian elections; documented findings in `milestones/M12_robustness_shift/results/scenario_data_search.md`.
- Created `ares_v2/src/ares/synth/`:
  - `facts.py`: Extracted atomic facts from Wikipedia/ASDMA/ECI snapshots with Wikidata QIDs and GeoNames coordinates.
  - `perturb.py`: Pure deterministic perturbation engine (8 operators).
  - `validate.py`: Validation suite enforcing slot preservation, dual NLI filters, safety/slur checks, and deduplication.
  - `generate_bench.py`: Generated and froze ARES-Synth v1 ($n=1858$, SIMULATED) at `data/synthetic/ares_synth_v1/` and `ares_v2/data_manifests/ares_synth_v1.csv` with manifest hash.
- Created `ares_v2/src/ares/robustness/synth_eval.py`:
  - Test-only evaluation suite for frozen models on ARES-Synth v1.

### 4. Temporal External Validation Track T11
- Drafted and committed `milestones/M12_robustness_shift/PREREG_T11.md` establishing the post-freeze prospective temporal validation protocol on IFCN ClaimReview fact-checks with strict evidence timestamp constraint ($t_{\text{evidence}} \le t_{\text{claim}}$).

### 5. Master Pipeline Execution & Deliverables Generation
- Built `ares_v2/scripts/run_m12_evaluation.py`.
- Fixed headless Tkinter Win32 GUI error by enforcing `matplotlib.use("Agg")`.
- Refined adversarial test feature construction to sparse active token substitution.
- Executed full evaluation pipeline in 205.45 seconds:
  - Generated all 10 CSV tables in `milestones/M12_robustness_shift/tables/`.
  - Generated all 4 publication figures in `milestones/M12_robustness_shift/figures/`.
  - Logged test set access to `milestones/test_set_access_log.csv`.
  - Validated and saved machine-readable results to `milestones/M12_robustness_shift/results/M12-robustness-shift-s42.json`.

### 6. Verification & Ledger Reconciliation
- Built unit test suites `ares_v2/tests/test_synth.py` and `ares_v2/tests/test_robustness.py`.
- Executed full pytest suite across repository: 96 of 96 tests passed (100% pass rate).
- Reconciled all 15 target ledger claims in `plan/claims_ledger.csv` and created `milestones/M12_robustness_shift/ledger_delta.csv`.
- Wrote `milestones/M12_robustness_shift/AUDIT.md` and `README.md`.
