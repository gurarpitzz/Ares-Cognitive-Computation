# Milestone M15 Pre-Registration — Release, Reproducibility Package, References & Manuscript Reconciliation

**Milestone:** M15 — Release, Reproducibility Package, References & Manuscript Reconciliation  
**Date:** 2026-10-14  
**Status:** PRE-REGISTERED  
**Target Ledger Claims:** `C024`, `C025`, `C026`, `C027`, `C030`, `C076`, `C079`, `C090`, `C092`, `C099` + complete close-out of all 99 rows in `plan/claims_ledger.csv`.  
**Execution Environment:** HP OMEN 16-wd0xxx (NVIDIA RTX 4060 Laptop GPU 8 GB GDDR6, Intel Core i7-13620H, 16 GB DDR5 RAM, Windows 11 Home 24H2, WSL2 Ubuntu)  
**Primary Deliverables:**
- Clean-Room Smoke Reproduction Report (`results/cleanroom_report.json`)
- Verified BibTeX Database & Audit Table (`references.bib`, `tables/table_reference_verification.csv`)
- Comprehensive Related Work Capability Matrix (`tables/table_related_work_comparison.csv`)
- Retrospective Trace Analysis for Use Cases (`tables/table_retrospective_traces.csv`)
- Comprehensive Manuscript Reconciliation Document (`manuscript_reconciliation.md`)
- Point-by-Point Response to Reviewers v2 (`response_to_reviewers_v2.md`)
- Final Meta-Audit Report (`META_AUDIT.md`)
- Complete Ledger Reconciliation & Summary (`ledger_final_summary.md`, `ledger_delta.csv`, updated `plan/claims_ledger.csv`)
- Release Artifact Specifications & Reproducibility Guide (`docs/reproduce.md`, `data_availability.md`)
- Blocked Gate Declaration for Fabricated Claims (`milestones/M15_release/BLOCKED.md`)

---

## 1. Experimental Plan & Scope

Milestone M15 constitutes the final release synthesis and manuscript reconciliation phase of the ARES 2.0 platform. It freezes all code, models, manifests, and results from M00–M14, audits every bibliographic citation, resolves all 10 remaining unverified ledger claims, regenerates all manuscript figures and tables with vector precision, and drafts an exhaustive section-by-section rewrite protocol and reviewer response:

1. **Clean-Room Smoke Reproduction (M15.1):**
   - Execute an isolated verification run across the test harnesses of all milestones (M00–M14).
   - Re-evaluate 3 randomly selected headline experiments (`seed = 20261013`) and compare measured outputs with stored results within strict numerical tolerance ($|\Delta| \le 0.005$).
   - Confirm hash integrity of all data manifests and generator outputs.
2. **Release Artifacts & Open-Source Infrastructure (M15.2, Claim C076):**
   - Provide complete Data & Code Availability statement detailing open benchmarks (LIAR, Constraint, CoAID, X-FACT, FEVER, AVeriTeC, FF++, Celeb-DF, ASVspoof, PHEME, TwiBot-20), restricted data access procedures, model card specifications, and Zenodo DOI reservation (`10.5281/zenodo.10826492`).
   - Create unified reproducibility guide (`docs/reproduce.md`) mapping exact terminal commands, seeds, hardware profiles, and runtimes.
3. **Comprehensive Ledger Close-Out (M15.3):**
   - Resolve every remaining `UNVERIFIED` claim in `plan/claims_ledger.csv`.
   - Achieve 100% terminal status across all 99 ledger rows: `REPRODUCED`, `REVISED`, `CORRECTED-THEORY`, `VERIFIED`, or `REMOVE`.
   - Export `ledger_final_summary.md` detailing status counts, delta transitions, and justifications.
4. **Reference Verification & Bibliography Overhaul (M15.6, Claim C092):**
   - Audit all 50 manuscript citations against Crossref and Semantic Scholar APIs.
   - Correct all 14 identified erroneous citations (e.g. Schuster et al. $\to$ Graves et al. 2013; Grinberg et al. 2020 $\to$ 2019; Gordon Pennycook naming; FEQA attribution; etc.).
   - Expand bibliography to 65 verified entries, adding required citations for modern baselines, architectures, and datasets.
   - Resolve every inline citation placeholder `[?]` flagged by Reviewers 2 and 3.
5. **Related Work Systematic Comparison & Scoped Novelty (M15.7, Claim C079):**
   - Build a structured capability comparison table benchmarking ARES 2.0 against 10 prominent misinformation detection frameworks across 8 functional dimensions.
   - Excise unsubstantiated "first system to unify" assertions, replacing them with precise, defensible claims regarding evidence-grounded multimodal verification, mutation-aware claim memory, and decision-theoretic abstention.
6. **Resolution of Real-World Impact Claims (Claims C024, C025, C026, C027, C030, C090):**
   - Formally block and mark **REMOVE** for claims asserting ungrounded real-world deployment outcomes (panic traffic reduction, military intelligence alerts, 13,000+ bot removals, public trust surveys) via [`BLOCKED.md`](file:///d:/ares/milestones/M15_release/BLOCKED.md).
   - Reframe Ladakh border and Assam flood scenarios as controlled grounded-synthetic evaluations (ARES-Synth v1, M12/M16) explicitly tagged **SIMULATED**.
   - Provide empirical retrospective verification traces on real historical claims to demonstrate real-system telemetry without fabricated external outcomes.
7. **Manuscript Reconciliation Protocol (M15.8):**
   - Produce `manuscript_reconciliation.md` with explicit section-by-section rewrite instructions covering Title, Abstract, Introduction, Related Work, Architecture, Verification Layers, Theory, Systems Performance, Ethics Safeguards, Limitations, and Appendices.
   - Enforce the fundamental rule: **No number may appear in the revised manuscript unless it is grounded in the ledger with an empirical evidence path.**
8. **Point-by-Point Response to Reviewers v2 (M15.9):**
   - Draft `response_to_reviewers_v2.md` systematically resolving all critiques from Reviewers 1, 2, and 3, citing specific file lines, tables, and commits.
9. **Final Meta-Audit (M15.10):**
   - Independently sample 20 numbers from `manuscript_reconciliation.md` and trace each back to raw results JSON.
   - Execute the entire project test suite (110+ unit tests) and verify zero leakage, zero ungrounded numbers, and zero unverified references.

---

## 2. Red-Flag Anti-Cheat Gates (RF1–RF6)

| Gate ID | Requirement | Verification Protocol | Pass Criteria |
| :--- | :--- | :--- | :--- |
| **RF1** | **Traceability of Reconciled Numbers** | Every number appearing in `manuscript_reconciliation.md` must link to an entry in `plan/claims_ledger.csv` with status `REPRODUCED`, `REVISED`, or `VERIFIED`. | 0 ungrounded numbers. |
| **RF2** | **Zero Unverified References** | Every entry in `references.bib` must be verified with DOI, author list, venue, and year. Zero unresolved `[?]` markers. | 100% reference verification. |
| **RF3** | **Zero Restricted Data in Public Manifests** | Check all release manifests, model cards, and documentation for unauthorized distribution of proprietary data. | 0 copyright/license breaches. |
| **RF4** | **Clean-Room Smoke Reproducibility** | Execute verification harnesses across milestones; compare 3 sampled headline experiments with stored JSON within $|\Delta| \le 0.005$. | 100% smoke pass rate. |
| **RF5** | **Zero Promotional Language in Text** | Audit Abstract and Conclusion in `manuscript_reconciliation.md` for hype adjectives ("unprecedented", "revolutionary", "flawless"). | Strict factual, neutral academic prose. |
| **RF6** | **Mandatory SIMULATED Qualifier** | Any metric derived from synthetic scenarios (ARES-Synth, queueing simulations) must explicitly bear the `SIMULATED` tag. | 0 un-flagged synthetic claims. |

---

## 3. Pre-Registered Hypotheses

- **H15.1 (Ledger Completeness):** 100% of claims in `plan/claims_ledger.csv` (all 99 rows) achieve terminal verification status (`status != 'UNVERIFIED'`), with 0 orphan rows.
- **H15.2 (Bibliographic Integrity):** 100% of references in `references.bib` (65/65) match verified scholarly publications, with all 14 manuscript errors corrected and all inline `[?]` placeholders resolved.
- **H15.3 (Clean-Room Reproducibility):** Smoke tests across all milestone harnesses pass with 100% success, and 3 randomly audited headline metrics reproduce within tolerance $|\Delta| \le 0.005$.
- **H15.4 (Meta-Audit Traceability):** A random sample of 20 numbers drawn from `manuscript_reconciliation.md` achieves 100% end-to-end traceability to valid `ExperimentResult` JSON files.

---

## 4. Decision Rules & Target Ledger Action Matrix

| Claim ID | Section | Original Claim | Status | Action in M15 |
| :--- | :---: | :--- | :---: | :--- |
| **C024** | XIII-A | 18 min alert; -47% panic traffic | **REMOVE** | Blocked via `BLOCKED.md`. Remove real-world traffic reduction claim; reframe Assam flood as grounded-synthetic benchmark. |
| **C025** | XIII-B | Classified alert to military intelligence | **REMOVE** | Blocked via `BLOCKED.md`. Delete military intelligence alert claim entirely; reframe Ladakh border as grounded-synthetic benchmark. |
| **C026** | XIII-C | 94.7% confidence; PSA launched 6h earlier | **REVISED** | Remove PSA timing claim; report calibrated posterior probability $P(Y=\text{false}\mid x)=0.941$ from retrospective trace. |
| **C027** | XIII-D | 13,000+ bot posts removed | **REMOVE** | Blocked via `BLOCKED.md`. Remove platform moderation claim; cite offline bot detection results from M09. |
| **C030** | XIII-E | +62% trust gain (survey n=4000) | **REMOVE** | Blocked via `BLOCKED.md`. Remove ungrounded human survey claim; require IRB approval for future human perception studies. |
| **C076** | Data | All data included in article | **REVISED** | Comprehensive Data & Code Availability statement with repository links, Zenodo DOI, and license specifications. |
| **C079** | II/XXIV | First system to unify all dimensions | **REVISED** | Systematic comparison table across 10 platforms; scoped, defensible novelty statement based on verified integration. |
| **C090** | XII-B | 5-stage MTurk + IFCN crowdsourcing | **REVISED** | Remove claim of author-conducted crowdsourcing; document true provenance of peer-reviewed benchmarks and ARES-Synth. |
| **C092** | Refs | 50 references, verified | **REVISED** | Overhaul bibliography: correct 14 errors, expand to 65 verified entries, resolve all inline `[?]` markers. |
| **C099** | Guide | Legacy evaluate.py 50-sample command | **REVISED** | Complete reproducibility guide (`docs/reproduce.md`) mapping exact commands, configs, runtimes, and seeds for all tracks. |
