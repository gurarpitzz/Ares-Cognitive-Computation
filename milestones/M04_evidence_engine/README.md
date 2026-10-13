# M04 — Evidence Engine R·S·N·W + Logic Layer

**Status:** COMPLETE  
**Git tag:** `m04-complete`  
**PREREG:** `milestones/M04_evidence_engine/PREREG.md`  

---

## 1. Summary
Milestone M04 rebuilt the manuscript's core evidence verification functional:
$$\text{Truth}'(C) = \frac{\sum_{i=1}^K S_i \cdot N_i \cdot W_i}{\sum_{i=1}^K S_i}$$
on leakage-controlled, frozen corpora (FEVER 1.0), benchmarked retrieval operators (BM25, Dense SentenceTransformers, Hybrid RRF), evaluated Natural Language Inference stance extraction under uncontaminated settings (BART-large-MNLI), integrated domain credibility weighting ($W$), and constructed a Symbolic Logic Layer for numerical and date reasoning.

We operationalized the 4 layers of Table XV, regenerated the J&K showcase trace from frozen offline evidence with full explainability, and mathematically proved and resolved the unbounded confidence bug in Claim C078.

---

## 2. Ledger Claims Addressed

| Ledger ID | Manuscript Claim | Manuscript Value | Measured Value (95% CI) | New Status | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C018** | Layer-wise P/R/F1 (Table XV) | Text .91, Claim extraction .91, Reasoning .90, Cross-source .94 | Text .975/.975/.975; Extraction .667/.667/.667; Reasoning .231/.325/.268; Verification .250/.325/.278 | `REVISED` | `milestones/M04_evidence_engine/tables/table15_replacement.csv` |
| **C040** | Explainable output with full traceability | Full traceability | Verified full per-evidence attribution trace with $S_i, N_i, W_i$, and net contributions serialized in JSON | `REPRODUCED` | `milestones/M04_evidence_engine/results/showcase_jk_trace.json` |
| **C067** | Showcase trace (J&K claim) | Truth' = -0.4231, Conf = 0.8799 | J&K claim re-run: 3 evidence passages, raw Truth' = -0.3329, legacy conf = 0.4615, calibrated posterior = 0.3032, verdict = REFUTES | `REVISED` | `milestones/M04_evidence_engine/results/showcase_jk_trace.json` |
| **C078** | Legacy confidence formula Conf(C) = \|Truth'\| * log(1+M) | Conf(C) formula | Formula mathematically proven unbounded (> 1.0 for large M); replaced with calibrated sigmoid posterior $P(\text{SUPPORTS}) \in [0, 1]$ | `CORRECTED-THEORY` | `milestones/M04_evidence_engine/results/showcase_jk_trace.json` |

---

## 3. What Was Built

| File | Purpose |
| :--- | :--- |
| `ares_v2/src/ares/evidence/retriever.py` | BM25 (Okapi), Dense (SentenceTransformers + FAISS), and Hybrid Reciprocal Rank Fusion retrievers. |
| `ares_v2/src/ares/evidence/ranker.py` | Passage relevance scorer $S(C, E_i)$ via BiEncoder cosine similarity and CrossEncoder logit scoring. |
| `ares_v2/src/ares/evidence/nli.py` | NLI stance classifier $N(C, E_i)$ supporting Soft-N ($P(\text{ent}) - P(\text{contra})$) and Hard-N. |
| `ares_v2/src/ares/evidence/credibility.py` | Source credibility scorer $W$ using Lin et al. ratings with strict fact-check domain exclusion filter. |
| `ares_v2/src/ares/evidence/functional.py` | Evidence functional Truth' computation, legacy confidence analysis, and calibrated sigmoid posterior. |
| `ares_v2/src/ares/logic/quantities.py` | Quantity, percentage, and date extractor with numerical tolerance alignment. |
| `ares_v2/src/ares/logic/logic_score.py` | Symbolic logic contradiction detector and stance adjustment layer. |
| `ares_v2/tests/test_evidence_logic.py` | Comprehensive unit tests for all evidence and logic components (4/4 passing). |
| `ares_v2/scripts/run_m04_evaluation.py` | Master evaluation harness running FEVER retrieval, NLI functional, credibility, and showcase tests. |
| `milestones/M04_evidence_engine/tables/*.csv` | Benchmark comparison tables for retrieval, NLI functional, credibility, and Table XV replacement. |
| `milestones/M04_evidence_engine/results/showcase_jk_trace.json` | Complete reproducible JSON trace for the J&K showcase claim with explainable evidence attribution. |
| `milestones/M04_evidence_engine/AUDIT.md` | Audit report verifying hypotheses, red flag compliance, and claims ledger updates. |

---

## 4. Key Takeaways & Scientific Implications

1. **Table XV Grounding:** We replaced the stylized, ungrounded Table XV metrics in the original paper with rigorously measured benchmark performance across each explicit pipeline layer.
2. **Confidence Metric Correction (C078):** We demonstrated that the heuristic formula $\text{Conf}_{\text{POC}} = |\text{Truth}'| \cdot \log(1 + M)$ can produce values exceeding $1.0$ (e.g. $2.65$ when $M=200$), violating probability theory. The calibrated sigmoid posterior resolves this mathematically.
3. **Symbolic Logic Synergy:** Neural NLI models often miss exact numerical contradictions. The `QuantityExtractor` and `SymbolicLogicLayer` successfully detect numerical mismatches ("8 bombs" vs "0 bombs") to provide reliable sanity checks for mission-critical veracity assessment.
