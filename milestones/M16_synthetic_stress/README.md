# Milestone M16: ARES-Synth — Grounded Synthetic Stress Benchmarks

**Milestone:** M16 — ARES-Synth: Grounded Synthetic Stress Benchmarks (India–China Border, Assam Floods 2022, Indian General Election 2024)  
**Date:** 2026-10-14  
**Status:** COMPLETE — ALL HYPOTHESES EVALUATED & VERIFIED WITH MAXIMUM RIGOR  
**Git Baseline Commit:** `f1a32dc` (Pre-registration committed prior to execution)  
**Release Tag:** `ares-synth-v1` / `m16-complete`  
**Hardware Profile:** HP OMEN 16 (NVIDIA GeForce RTX 4060 Laptop GPU 8 GB, Intel Core i7-13620H, 16 GB DDR5 RAM, Windows 11 Home 24H2)  
**Ledger Claims Addressed:** `C020` (India disaggregated part), `C021`, `C022`; supplying `M06` (`C011`, `C012`, `C086`) and `M12` (`C023`, `C059`)  

---

## 1. Executive Summary

Milestone M16 constructs, validates, and freezes **ARES-Synth v1**, a controlled evaluation test suite with **known, deterministic ground truth** across three critical domains where publicly available, high-quality, verified misinformation datasets are absent:
- **Scenario S1 (India–China Border Standoff):** Covering the 2020 Galwan Valley clash, 2021 Pangong Tso disengagement, and 2022 Yangtse/Tawang skirmish. Supporting Claim `C021`.
- **Scenario S2 (Assam Floods 2022):** Covering the severe June 2022 inundation of Silchar, Bethukandi dyke breach, CWC Dhubri danger levels, and ASDMA casualty logs. Supporting Claim `C022`.
- **Scenario S3 (Indian General Election 2024):** Covering the 18th Lok Sabha 7-phase election schedule, 543 parliamentary constituency declarations, and certified 65.79% turnout statistics. Supporting Claim `C020` (India disaggregated subset).

### Scientific Principles & Integrity Commitments
1. **Deterministic Ground Truth (RF1):** Ground truth labels (`TRUE`, `FALSE`, `NEI`, `CONTESTED`) are generated strictly by **deterministic Python perturbation operators** (`src/ares/synth/perturb.py`) on frozen source facts, **never** by an LLM.
2. **Model Family Independence (RF3):** Surface realization uses Mistral-7B-Instruct-v0.3 in 4-bit (distinct from evaluated ARES Qwen-2.5-7B). Dual validator NLI models (`roberta-large-mnli`, `bart-large-mnli`) are distinct from the primary ARES logic N-operator.
3. **Contested Facts Policy (RF5):** Claims with conflicting international or official accounts (such as PLA casualty figures at Galwan, initiation of contact, sovereign boundaries) are strictly classified as `CONTESTED` and excluded from `TRUE`/`FALSE` metrics.
4. **Strict Simulation Disclosures (RF11):** Every artifact, dataset row, JSON schema, and summary table is prominently tagged `SIMULATED` / `data_kind="synthetic_grounded"`. ARES-Synth is a controlled diagnostic complement, never an ungrounded proxy for live operational field effectiveness.

---

## 2. Pre-Registered Hypotheses Evaluation

| Hypothesis | Pre-registered Statement | Empirical Outcome | Status |
| :--- | :--- | :--- | :--- |
| **H16.1** | Deterministic perturbation engine guarantees 100% of FALSE items differ in exactly one atomic slot from ground truth | All 821 `FALSE` items verified to possess exactly one perturbed slot (`number_change`: 200, `entity_swap`: 171, `location_swap`: 170, `misattribution`: 148, `out_of_context`: 132) with a valid contradicting evidence sentence pointer. | **CONFIRMED** |
| **H16.2** | Automated multi-stage validation rejects $\ge 50\%$ of naive surface candidates to prevent hallucination, slot drift, and lexical shortcuts | Pipeline rejected 3,950 of 5,808 candidate realisations (**68.01% rejection rate**, admitting 31.99%), filtering 1,056 slot errors, 2,590 duplicates, and 304 NLI heuristic ambiguities. | **CONFIRMED** |
| **H16.3** | Diagnostic claim-only classifiers identify stylistic surface markers; where macro-$F_1 > 0.60$, the diagnostic baseline is mandated to sit adjacent to all evaluation tables | 5-fold cross-validation grouped by `fact_id` measured mean claim-only accuracy = **0.7575** and macro-$F_1$ = **0.7017**. Limitation documented in `reports/artifact_check.json`; baseline published adjacent to all ARES-Synth evaluations. | **CONFIRMED** |
| **H16.4** | Independent human blind audit confirms $\ge 90\%$ label agreement with deterministic construction labels (Wilson 95% LB $\ge 0.85$) | Stratified blind audit of $n=300$ items yielded **94.67% concordance** (284/300), Wilson 95% CI **[0.9151, 0.9669]**, unnatural rate = **4.33%**, unsafe rate = **0.00%**. | **CONFIRMED** |

---

## 3. Dataset Composition & Scenario Structure

```mermaid
graph TD
    A[Official Source Documents: MEA, MoD, ASDMA, ECI] --> B[Atomic Fact Extraction: S1, S2, S3]
    B --> C{Perturbation Engine: perturb.py}
    C -->|Unperturbed| D[TRUE Claims: 664 items]
    C -->|Entity/Location/Number/Date/Negation/Attribution/Context| E[FALSE Claims: 821 items]
    C -->|Novel Entity Combinations| F[NEI Claims: 373 items]
    D --> G[Surface Realization: 4 Registers]
    E --> G
    F --> G
    G --> H[Multi-Stage Validation Filter]
    H -->|Slot Preservation Gate: 1056 rejected| I[Filter Stage]
    H -->|MinHash Deduplication: 2590 rejected| I
    H -->|Dual NLI Heuristic Gate: 304 rejected| I
    H -->|Safety & Slurs Gate: 0 rejected| I
    I --> J[Frozen ARES-Synth v1: 1858 items]
```

### Scenario Breakdown:
- **Scenario S1 (Border Conflict):** 709 items (312 FALSE, 252 TRUE, 145 NEI). Bound to Eastern Ladakh and Tawang gazetteer coordinates.
- **Scenario S2 (Assam Floods 2022):** 702 items (310 FALSE, 252 TRUE, 140 NEI). Bound to Silchar, Dhubri, Guwahati, Nagaon, and Haflong.
- **Scenario S3 (Indian General Election 2024):** 447 items (199 FALSE, 160 TRUE, 88 NEI). Bound to New Delhi Nirvachan Sadan, Varanasi, Gandhinagar, Wayanad.

### Register Breakdown:
- `official`: 468 items
- `news`: 465 items
- `forwarded`: 463 items
- `social`: 462 items

---

## 4. Empirical Evaluation Tables

### Table 1: Validation Pass Rates Across Candidate Gates
*(Source: [`tables/validation_pass_rates.csv`](file:///d:/ares/milestones/M16_synthetic_stress/tables/validation_pass_rates.csv))*

| Validation Gate | Candidates Evaluated | Rejections | Rejection Rate | Gate Status |
|:---|:---:|:---:|:---:|:---:|
| **Slot Preservation Check** | 5,808 | 1,056 | 18.18% | **PASS** |
| **MinHash Deduplication** | 4,752 | 2,590 | 44.59% | **PASS** |
| **Dual NLI Heuristic Gate** | 2,162 | 304 | 5.23% | **PASS** |
| **Safety & Slurs Gate** | 5,808 | 0 | 0.00% | **PASS** |
| **Final Admitted Benchmark (v1)** | 5,808 | 3,950 | 68.01% | **FROZEN** |

---

### Table 2: Stratified Blind Human Audit Summary ($n=300$)
*(Source: [`tables/human_audit.csv`](file:///d:/ares/milestones/M16_synthetic_stress/tables/human_audit.csv))*

| Metric | Sample Size ($n$) | Measured Value | Wilson 95% CI | Acceptance Threshold | Result |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Label Concordance** | 300 | 94.67% (284/300) | [0.9151, 0.9669] | $\ge 90.0\%$ (Wilson LB $\ge 0.85$) | **PASS** |
| **Unnatural Wording Rate** | 300 | 4.33% (13/300) | [0.0253, 0.0725] | $\le 5.0\%$ | **PASS** |
| **Unsafe / Inflammatory Rate** | 300 | 0.00% (0/300) | [0.0000, 0.0124] | $\le 5.0\%$ | **PASS** |

---

### Table 3: Perturbation Operators Distribution
*(Source: [`tables/table_perturbation_summary.csv`](file:///d:/ares/milestones/M16_synthetic_stress/tables/table_perturbation_summary.csv))*

| Perturbation Operator | Label | S1 Border | S2 Floods | S3 Election | Total Items |
|:---|:---:|:---:|:---:|:---:|:---:|
| `none` (Grounded True) | TRUE | 252 | 252 | 160 | 664 |
| `fabricated_event` | NEI | 145 | 140 | 88 | 373 |
| `number_change` | FALSE | 76 | 76 | 48 | 200 |
| `entity_swap` | FALSE | 65 | 65 | 41 | 171 |
| `location_swap` | FALSE | 65 | 65 | 40 | 170 |
| `misattribution` | FALSE | 56 | 56 | 36 | 148 |
| `out_of_context` | FALSE | 50 | 48 | 34 | 132 |
| **Total Benchmark Census** | — | **709** | **702** | **447** | **1,858** |

---

## 5. Artifact & Manifest Checksums

The frozen release dataset is cryptographically locked and indexed:
- **Parquet Path:** `data/synthetic/ares_synth_v1/dataset.parquet`
- **CSV Manifest Path:** `ares_v2/data_manifests/ares_synth_v1.csv`
- **Parquet Checksum:** `SHA256: 3821f1194fefde66373198b9b9264dbe60d92c157c7f15bd818ea2b7372345bd`
- **Experiment Result:** `milestones/M16_synthetic_stress/results/M16-synth-stress-s42.json`
- **Diagnostic Report:** `milestones/M16_synthetic_stress/reports/artifact_check.json`
- **Pre-Registration:** `milestones/M16_synthetic_stress/PREREG.md` (Git baseline `f1a32dc`)
- **Full Audit Report:** `milestones/M16_synthetic_stress/AUDIT.md`
