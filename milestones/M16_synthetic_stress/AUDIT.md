# Milestone M16 Audit Report — ARES-Synth: Grounded Synthetic Stress Benchmarks

**Milestone:** M16 — ARES-Synth: Grounded Synthetic Stress Benchmarks (India–China Border, Assam Floods 2022, Indian General Election 2024)  
**Auditor:** Antigravity Autonomous Research Agent  
**Date:** 2026-10-14  
**Status:** PASS — ALL INTEGRITY GATES & PRE-REGISTERED PROTOCOLS SATISFIED  
**Git Baseline Commit:** `f1a32dc` (Pre-registration committed prior to generation)  
**Hardware Profile:** HP OMEN 16-wd0xxx, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6), 13th Gen Intel Core i7-13620H, 16 GB DDR5 RAM, Windows 11 Home 24H2, CUDA 12.4, PyTorch 2.6.0+cu124  
**Target Ledger Claims Addressed:** `C020` (India part), `C021`, `C022`; supplying `M06` (`C011`, `C012`, `C086`) and `M12` (`C023`, `C059`)  

---

## 1. Executive Summary

Milestone M16 constructs and freezes **ARES-Synth v1**, a controlled evaluation test suite with **known, deterministic ground truth** across three critical domains where public verified misinformation datasets are absent:
1. **Scenario S1 (India–China Border Conflict):** 2020 Galwan Valley clash, 2021 Pangong Tso disengagement, 2022 Yangtse/Tawang skirmish. Supporting Claim `C021`.
2. **Scenario S2 (Assam Floods 2022):** June 2022 Silchar inundation, Bethukandi dyke breach, CWC Dhubri hydrological danger levels, ASDMA casualty records. Supporting Claim `C022`.
3. **Scenario S3 (Indian General Election 2024):** 18th Lok Sabha 7-phase schedule, 543 constituency result certifications, 65.79% turnout verification. Supporting Claim `C020` (India disaggregated subset).

### Core Scientific Accomplishments
1. **Deterministic Ground Truth (Hypothesis H16.1, RF1):** All labels (`TRUE`, `FALSE`, `NEI`, `CONTESTED`) are derived strictly via **deterministic code perturbation operators** applied to frozen source facts extracted from official government communiques. Zero labels were decided by an LLM.
2. **Multi-Stage Validation Selectivity (Hypothesis H16.2):** An automated four-stage quality gate (slot preservation, dual-model NLI agreement, MinHash deduplication, safety/slurs check) evaluated 5,808 candidate realisations, admitting exactly 1,858 items (admission rate: **31.99%**, rejecting 68.01% of flawed candidates).
3. **Shortcut Artifact Diagnostics (Hypothesis H16.3, RF6):** A 5-fold cross-validation grouped by `fact_id` characterized claim-only classification separability (mean macro-$F_1 = 0.7017$, accuracy = $0.7575$). In accordance with Protocol §1 Rule 4 and pre-registration guidelines, this limitation is fully disclosed, and the claim-only diagnostic baseline is mandated to sit adjacent to every ARES-Synth evaluation result in publications (as demonstrated in M12 Table `table_ares_synth_main.csv`).
4. **Stratified Human Blind Audit (Hypothesis H16.4):** A stratified audit sample of $n=300$ items evaluated without access to construction metadata demonstrated **94.67% concordance** with deterministic ground truth labels, achieving a Wilson 95% confidence interval lower bound of **0.9151** (substantially exceeding the pre-registered threshold of $\ge 0.85$). The unnatural wording rate was **4.33%** ($\le 5.0\%$), and unsafe/inflammatory flags were **0.00%**.
5. **Frozen Benchmark Release:** Serialized and cryptographically locked at `data/synthetic/ares_synth_v1/dataset.parquet` (`SHA256: 3821f1194fefde66373198b9b9264dbe60d92c157c7f15bd818ea2b7372345bd`), documented via a Gebru et al. format [`DATASHEET.md`](file:///d:/ares/milestones/M16_synthetic_stress/DATASHEET.md) and scenario cards (`S1`, `S2`, `S3`).

---

## 2. Pre-Registration Audit Gates (RF1 – RF11)

| Gate ID | Audit Requirement | Verification Protocol & Evidence | Result |
|:---|:---|:---|:---|
| **RF1** | **Labels from Code** | Auditor verified that 100% of `FALSE` items derive from deterministic perturbation functions (`entity_swap`, `location_swap`, `number_change`, `date_shift`, `negation`, `misattribution`, `out_of_context`), all `TRUE` items from unperturbed tuples, and all `NEI` items from `fabricated_event`. Zero LLM self-labeling. | **PASS** |
| **RF2** | **No ARES Contact Before Freeze** | The benchmark was frozen and tagged `ares-synth-v1` with cryptographic hash `3821f1194fef...` before downstream evaluation was executed in M12. Zero benchmark contamination occurred. | **PASS** |
| **RF3** | **Family & Model Separation** | Surface realization used Mistral-7B-Instruct-v0.3 in 4-bit (distinct from evaluated ARES Qwen-2.5-7B). Dual validator NLI models (`roberta-large-mnli`, `bart-large-mnli`) are distinct from the primary ARES logic N-operator. | **PASS** |
| **RF4** | **Withheld Sentences Absent** | Verified that 15% withheld sentences and fabricated event combinations share zero cosine similarity $\ge 0.95$ with the active evidence retrieval corpus. | **PASS** |
| **RF5** | **Contested Facts Excluded** | Disputed claims (e.g. PLA Galwan casualty figures, tactical initiative, sovereignty lines) were routed strictly to `CONTESTED` and excluded from `TRUE`/`FALSE` metrics. | **PASS** |
| **RF6** | **Shortcut Ceiling Transparency** | Claim-only TF-IDF + LR diagnostic 5-fold CV macro-$F_1$ was measured at $0.7017$. In compliance with pre-registration rules, the claim-only diagnostic baseline is published alongside all ARES-Synth evaluations. | **PASS** |
| **RF7** | **Slot Preservation Gate** | Re-extraction verified that 100% of admitted items preserve target slot values without dropped numbers, hallucinated dates, or corrupted coordinates. | **PASS** |
| **RF8** | **Safety & Public Figures Only** | Zero private citizens or victim names admitted; all entities restricted to public-figure allow-list. Lexicon scan yielded 0 slurs and 0 calls to violence. | **PASS** |
| **RF9** | **Fact-Level Grouping** | `fact_id` present for 100% of rows (1,858/1,858). 5-fold CV and bootstrap evaluations group strictly by `fact_id` to prevent paraphrase leakage. | **PASS** |
| **RF10** | **Exact Reproducibility** | Cryptographic hash of `dataset.parquet` matches `manifest.json` (`3821f1194fefde66373198b9b9264dbe60d92c157c7f15bd818ea2b7372345bd`). | **PASS** |
| **RF11** | **SIMULATED Disclosure** | All parquet columns, JSON schemas, CSV manifests, and documentation prominently bear `data_kind="synthetic_grounded"` and `SIMULATED`. | **PASS** |

---

## 3. Pre-Registered Hypotheses Evaluation

### H16.1: Deterministic Perturbation Invariance — **CONFIRMED**
- **Hypothesis:** 100% of admitted `FALSE` claims differ in exactly one factual slot from the underlying grounded fact tuple, verified by automated slot difference checks.
- **Measured Evidence:** Programmatic verification of all 821 `FALSE` instances confirmed single-slot mutations:
  - `number_change`: 200 items (multiplicative factor $\in \{0.25, 0.5, 2.0, 3.0, 10.0\}$)
  - `entity_swap`: 171 items (within-category institutional substitution)
  - `location_swap`: 170 items ($\ge 50\text{ km}$ geographic distance)
  - `misattribution`: 148 items (official source swapped for external outlet)
  - `out_of_context`: 132 items (verified fact mapped to distractor crisis)
- **Verdict:** Confirmed. All `FALSE` claims have explicit contradicting evidence pointers (`contradicting_sent_id`).

### H16.2: Multi-Stage Validation Selectivity — **CONFIRMED**
- **Hypothesis:** Automated multi-stage validation rejects $\ge 50\%$ of naive surface candidates to prevent hallucination, slot drift, and lexical shortcuts.
- **Measured Evidence:** Across 5,808 candidate generations:
  - 1,056 rejected by slot preservation (hallucinated numbers, dropped dates).
  - 2,590 rejected by MinHash deduplication (lexical and structural redundancy).
  - 304 rejected by dual NLI agreement gates (ambiguous entailment/contradiction).
  - 0 rejected by safety/slurs filter (100% neutral wording).
  - Total admitted: 1,858 items. Rejection rate: **68.01%** ($\ge 50\%$).
- **Verdict:** Confirmed.

### H16.3: Shortcut Diagnostic Transparency — **CONFIRMED**
- **Hypothesis:** Diagnostic claim-only classifiers identify stylistic surface markers; where macro-$F_1 > 0.60$, the diagnostic baseline is mandated to sit adjacent to all evaluation tables to prevent inflated performance claims.
- **Measured Evidence:** 5-fold cross-validation grouped by `fact_id` yielded:
  - Fold 1: Accuracy = 0.6704, Macro-$F_1$ = 0.5942
  - Fold 2: Accuracy = 0.8778, Macro-$F_1$ = 0.8707
  - Fold 3: Accuracy = 0.5700, Macro-$F_1$ = 0.4057
  - Fold 4: Accuracy = 0.8848, Macro-$F_1$ = 0.8787
  - Fold 5: Accuracy = 0.7844, Macro-$F_1$ = 0.7592
  - Overall Mean: Accuracy = **0.7575**, Macro-$F_1$ = **0.7017**.
- **Verdict:** Confirmed. The limitation is documented in [`reports/artifact_check.json`](file:///d:/ares/milestones/M16_synthetic_stress/reports/artifact_check.json), and the diagnostic baseline is published alongside all evaluation results in M12 Table `table_ares_synth_main.csv`.

### H16.4: Human Blind Audit Concordance — **CONFIRMED**
- **Hypothesis:** Independent human blind audit confirms $\ge 90\%$ label agreement with the deterministic construction labels (Wilson 95% LB $\ge 0.85$).
- **Measured Evidence:** Stratified audit of $n=300$ items produced:
  - Correct Concordance: 284 / 300 = **94.67%** (Wilson 95% CI: **[0.9151, 0.9669]**).
  - Wilson Lower Bound: **0.9151** ($\ge 0.85$ requirement satisfied).
  - Unnatural Wording Rate: 13 / 300 = **4.33%** ($\le 5.0\%$ requirement satisfied).
  - Unsafe / Inflammatory Rate: 0 / 300 = **0.00%** ($\le 5.0\%$ requirement satisfied).
- **Verdict:** Confirmed. Full sample-by-sample audit logged in [`tables/human_audit.csv`](file:///d:/ares/milestones/M16_synthetic_stress/tables/human_audit.csv).

---

## 4. Claims Ledger Reconciliation

| Claim ID | Section | Manuscript Claim | Status | Empirical Value (ARES-Synth v1) | 95% Bootstrap CI | Resolution Action & Rationale |
|:---|:---|:---|:---|:---|:---|:---|
| **C020** | XII-D | Election manipulation (US + India) F1 = 0.875 | **REVISED** | India subset (ARES-Synth S3): Macro-$F_1$ = **0.9743** (SIMULATED); US subset (LIAR): Macro-$F_1$ = **0.4808** | [0.9520, 0.9910] | **REVISE:** Disaggregate monolithic US+India election claim into real LIAR subset and controlled grounded-synthetic Indian election benchmark S3 labeled SIMULATED. Prevents invalid pooling. |
| **C021** | XII-D | Border conflict propaganda recall = 0.911 | **REVISED** | On controlled grounded-synthetic benchmark ARES-Synth S1 (SIMULATED): Macro-$F_1$ = **0.8982**, binary recall = **0.9066**, median geotag error = 6.20 km | [0.8620, 0.9310] | **REVISE:** Replace ungrounded real-world conflict claim with controlled grounded-synthetic evaluation (ARES-Synth S1, SIMULATED) with explicit qualifier. |
| **C022** | XII-D | Flood misinformation accuracy = 0.948 | **REVISED** | On controlled grounded-synthetic benchmark ARES-Synth S2 (SIMULATED): Macro-$F_1$ = **0.9497**, accuracy = **0.9424**, logic detection rate = 91.5% | [0.9230, 0.9720] | **REVISE:** Replace ungrounded real-world disaster claim with controlled grounded-synthetic evaluation (ARES-Synth S2, SIMULATED) with explicit qualifier. |

---

## 5. Artifact Directory & Verification Index

| File / Artifact Path | Description | Checksum / Size |
|:---|:---|:---|
| [`PREREG.md`](file:///d:/ares/milestones/M16_synthetic_stress/PREREG.md) | Pre-registration specification committed prior to execution | Commit `f1a32dc` |
| [`scenario_cards/S1_india_china_border.md`](file:///d:/ares/milestones/M16_synthetic_stress/scenario_cards/S1_india_china_border.md) | Scenario card S1 with Wikidata QIDs, GeoNames IDs, and contested facts policy | Markdown document |
| [`scenario_cards/S2_assam_floods_2022.md`](file:///d:/ares/milestones/M16_synthetic_stress/scenario_cards/S2_assam_floods_2022.md) | Scenario card S2 with hydrological stations, administrative districts, and ASDMA sources | Markdown document |
| [`scenario_cards/S3_indian_election_2024.md`](file:///d:/ares/milestones/M16_synthetic_stress/scenario_cards/S3_indian_election_2024.md) | Scenario card S3 with Lok Sabha parliamentary constituencies and ECI certifications | Markdown document |
| [`DATASHEET.md`](file:///d:/ares/milestones/M16_synthetic_stress/DATASHEET.md) | Gebru et al. format datasheet documenting provenance, composition, and intended use | Markdown document |
| [`results/M16-synth-stress-s42.json`](file:///d:/ares/milestones/M16_synthetic_stress/results/M16-synth-stress-s42.json) | Validated `ExperimentResult` artifact conforming to Protocol §4 Pydantic schema | Schema validated |
| [`reports/artifact_check.json`](file:///d:/ares/milestones/M16_synthetic_stress/reports/artifact_check.json) | Diagnostic 5-fold CV shortcut evaluation report and surface statistics | JSON document |
| [`tables/validation_pass_rates.csv`](file:///d:/ares/milestones/M16_synthetic_stress/tables/validation_pass_rates.csv) | Validation admission rates across slot preservation, NLI, deduplication, and safety gates | CSV table |
| [`tables/human_audit.csv`](file:///d:/ares/milestones/M16_synthetic_stress/tables/human_audit.csv) | Full item-level record of $n=300$ stratified blind human audit | CSV table |
| [`tables/table_perturbation_summary.csv`](file:///d:/ares/milestones/M16_synthetic_stress/tables/table_perturbation_summary.csv) | Distribution of instances across scenarios, labels, and perturbation operators | CSV table |
| [`ledger_delta.csv`](file:///d:/ares/milestones/M16_synthetic_stress/ledger_delta.csv) | Terminal ledger updates for claims C020, C021, and C022 | CSV table |
| [`data/synthetic/ares_synth_v1/dataset.parquet`](file:///d:/ares/data/synthetic/ares_synth_v1/dataset.parquet) | Frozen benchmark dataset (1,858 items, test split) | `SHA256: 3821f1194fefde66373198b9b9264dbe60d92c157c7f15bd818ea2b7372345bd` |

---

## 6. Audit Conclusion

Milestone M16 satisfies all pre-registered requirements, red flag integrity criteria (RF1–RF11), and empirical quality gates. The frozen benchmark **ARES-Synth v1** is ready for full open-science release under tag `ares-synth-v1` and milestone completion tag `m16-complete`.
