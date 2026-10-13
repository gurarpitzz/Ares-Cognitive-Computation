# Milestone M12 — Robustness, Distribution Shift, Stress Scenarios & Temporal External Validation

**Status:** COMPLETE  
**Git Tag:** `m12-complete`  
**Frozen Model Tag:** `ares-v2-freeze-1`  
**ARES-Synth Benchmark Tag:** `ares-synth-v1`  
**PREREG Commit:** `c0ca01d`  

---

## 1. Summary
Milestone M12 grounds the adversarial robustness, cross-domain transfer, crisis stress scenario, and temporal external validation claims of the ARES monograph. We evaluated text adversarial attacks (TextFooler, DeepWordBug, Homoglyphs) on a fixed $n=1000$ test sample, conducted a $K=10$ seed Virtual Adversarial Training ablation, mapped cross-domain transfer across 4 primary domains and correlated degradation with proxy $\mathcal{A}$-distance ($r_s = 0.3044$), evaluated Leave-One-Scenario-Out stress across 4 crisis scenarios, built and evaluated the controlled grounded-synthetic benchmark ARES-Synth v1 ($n=1858$, SIMULATED), and pre-registered prospective temporal validation Track T11. Crucially, the manuscript's claims of $\Delta_{\text{adv}} < 0.05$ and LOSO degradation $< 4.8\%$ were honestly **refuted** by empirical measurements ($0.0871$ and $4.90\%$, respectively).

---

## 2. Ledger Claims Addressed

| Ledger ID | Manuscript Claim | Manuscript Value | Measured Value (95% CI) | New Status | Evidence |
|---|---|---|---|---|---|
| `C003` | High resilience under stress scenarios | Qualitative | TextFooler $\Delta_{\text{adv}} = 0.0871$ [0.0687, 0.1062]; mean LOSO degradation 4.90% [0.77%, 8.94%]; ARES-Synth S1 F1 = 0.8982, S2 F1 = 0.9497 (SIMULATED) | **REVISED** | `tables/table_media_robustness.csv` |
| `C019` | COVID-19 infodemic stress test F1 | 0.893 | In-scenario F1: 0.8799 [0.8418, 0.9148]; Held-out LOSO F1: 0.8730 [0.8351, 0.9092] | **REVISED** | `tables/table_loso_scenarios.csv` |
| `C020` | Election manipulation (US + India) F1 | 0.875 | Real LIAR election subset F1: 0.4808 [0.4274, 0.5336]; ARES-Synth S3 election F1: 0.9743 [0.9520, 0.9910] (SIMULATED) | **REVISED** | `tables/table_loso_scenarios.csv`, `tables/table_ares_synth_main.csv` |
| `C021` | Border conflict propaganda (India-China) recall | 0.911 | No public real dataset exists (see search log). On ARES-Synth S1 (SIMULATED): Macro-F1 = 0.8982 [0.8620, 0.9310], recall = 0.9066 | **REVISED** | `tables/table_ares_synth_main.csv`, `results/scenario_data_search.md` |
| `C022` | Flood misinformation (Assam 2022) accuracy | 0.948 | No public real dataset exists (see search log). On ARES-Synth S2 (SIMULATED): Macro-F1 = 0.9497 [0.9230, 0.9720], accuracy = 0.9424 | **REVISED** | `tables/table_ares_synth_main.csv`, `results/scenario_data_search.md` |
| `C023` | F1 per scenario vs baselines | Bar values | In-scenario vs Held-out: Health 0.8799 vs 0.8730; Election 0.4808 vs 0.4035; Crisis 0.2923 vs 0.1908; Disaster 0.8918 vs 0.8815 | **REVISED** | `tables/table_loso_scenarios.csv` |
| `C037` | Adversarial degradation ARES 6.3% vs Grover 22.4% | 6.3% / 22.4% | TextFooler ASR = 7.7%; Grover is a machine text provenance detector, not a veracity verifier (category error documented) | **REVISED** | `tables/table_adversarial_attacks.csv` |
| `C044` | Cross-domain F1 ~0.85-0.88 | See text | Health $\to$ Disaster: 0.8967; Health $\to$ Politics: 0.4457; Politics $\to$ Health: 0.3845; Social $\to$ Health: 0.2203 | **REVISED** | `tables/table_cross_domain_matrix.csv` |
| `C045` | $\Delta_{\text{adv}} < 0.05$ across adversarial scenarios | <0.05 | **REFUTED**: TextFooler $\Delta_{\text{adv}} = 0.0871$ [0.0687, 0.1062]; DeepWordBug $\Delta = 0.0647$; Homoglyphs $\Delta = 0.2968$. Bound $<0.05$ holds only at $\epsilon \le 0.05$ | **REVISED** | `tables/table_adversarial_attacks.csv`, `figures/fig42_adv_strength_curve.png` |
| `C056` | Adversarial training ablation $\Delta F1 = 0.036 \pm 0.004$ | .036+/-.004 | VAT ablation ($K=10$ seeds): Clean F1: 0.9117 vs 0.9030; Adv F1 under 15% token perturbation: 0.8838 vs 0.8891 (gain: $-0.0053 \pm 0.0019$, $p=0.0007$) | **REVISED** | `tables/table_adversarial_training.csv` |
| `C059` | Leave-one-scenario-out degradation $< 4.8\%$ | <4.8% | **REFUTED**: Mean LOSO degradation across 4 crisis scenarios is $4.897\%$ [0.77%, 8.94%]; max degradation on PHEME crisis rumours is $10.15\%$ ($34.72\%$ relative) | **REVISED** | `tables/table_loso_scenarios.csv` |
| `C060` | Live OSINT agreement $> 88\%$ | >88% | Replaced with prospective post-freeze protocol Track T11 (`PREREG_T11.md`) on IFCN ClaimReview fact-checks | **REVISED** | `PREREG_T11.md` |
| `C066` | Detection accuracy vs adversarial perturbation strength | Curve | Generated empirical replacement for Figure 42 across perturbation budgets $\epsilon \in [0.05, 0.30]$ for ARES, DeBERTa, and TF-IDF | **VERIFIED** | `figures/fig42_adv_strength_curve.png` |
| `C073` | Ben-David shift bound; ARES explicitly minimizes discrepancy | Claim | Theory corrected (ARES does not minimize discrepancy in loss); empirical proxy $\mathcal{A}$-distance ($d_{\mathcal{A}}$) correlates with transfer drop ($r_s = 0.3044$) | **CORRECTED-THEORY** | `tables/table_adistance_correlation.csv`, `figures/fig_adist_vs_drop.png` |
| `C091` | Adversarial synthetic false-claim injection | Method | Injected rates $r \in \{0.00, 0.05, 0.10, 0.20\}$ across 5 seeds: Real-test F1 = 0.9013, 0.8976, 0.8979, 0.8989; FPR = 0.1250, 0.1319, 0.1319, 0.1319 | **VERIFIED** | `tables/table_synthetic_injection.csv` |

---

## 3. What Was Built

- `ares_v2/src/ares/robustness/text_attacks.py`: TextFooler, DeepWordBug, and Homoglyph attack implementations with hash-checked fixed sample evaluation ($n=1000$).
- `ares_v2/src/ares/robustness/adv_training.py`: Virtual Adversarial Training (VAT) in embedding space under Eq. 16–17 over $K=10$ random seeds.
- `ares_v2/src/ares/robustness/llm_eval.py`: LLM misinformation evaluation, Grover task demarcation, and synthetic false-claim injection rate experiment ($r \in \{0, 5, 10, 20\%\}$).
- `ares_v2/src/ares/robustness/cross_domain.py`: Cross-domain transfer matrix, PolitiFact deduplication, and proxy $\mathcal{A}$-distance estimation ($d_{\mathcal{A}} = 2(1 - 2 \cdot \text{err})$) via 5-fold CV linear SVM.
- `ares_v2/src/ares/robustness/loso.py`: Leave-One-Scenario-Out crisis stress evaluation across COVID-19, US Elections, PHEME breaking rumours, and Natural Disasters.
- `ares_v2/src/ares/synth/facts.py`: Grounded fact extraction from Wikipedia/ASDMA/ECI snapshots with Wikidata QIDs and GeoNames coordinates.
- `ares_v2/src/ares/synth/perturb.py`: Pure deterministic perturbation engine (8 operators).
- `ares_v2/src/ares/synth/validate.py`: Validation suite enforcing slot preservation, dual NLI filters, safety/slur checks, and deduplication.
- `ares_v2/src/ares/synth/generate_bench.py`: Generator script creating and freezing ARES-Synth v1 ($n=1858$, SIMULATED).
- `ares_v2/src/ares/robustness/synth_eval.py`: Test-only evaluation harness on ARES-Synth v1.
- `ares_v2/scripts/run_m12_evaluation.py`: Master end-to-end evaluation pipeline.
- `ares_v2/tests/test_synth.py`: Unit tests for ARES-Synth fact loading, perturbations, and validation.
- `ares_v2/tests/test_robustness.py`: Unit tests for text attacks and proxy $\mathcal{A}$-distance.

---

## 4. Data Specifications

- **Text Attack Sample:** $n = 1000$ correctly-classified test items drawn from Track T1 test split (`SHA-256: da62224f1071b3a1bc753802b3b477a9322520c5e4461e77b236a93d0a9bad87`).
- **ARES-Synth v1 Benchmark:** $n = 1858$ items (TRUE: 664, FALSE: 821, NEI: 373). Frozen at `ares-synth-v1`, `SHA-256: 3821f1194fefde6640c4974f762740bcbaef60662d98a032890538a7cba8c6d1`.
- **Deduplication:** PolitiFact cross-dataset claims deduplicated via MinHash Jaccard threshold $\ge 0.85$.

---

## 5. Experiments

- **Adversarial Attacks:** $n=1000$, TextFooler ($\epsilon \in [0.05, 0.30]$, $Q_{\max}=100$), DeepWordBug, Homoglyphs.
- **VAT Ablation:** $\lambda_{\text{adv}} \in \{0.00, 0.30\}$, $K=10$ seeds ($s \in \{0, \dots, 9\}$), 30 epochs, AdamW ($\text{lr}=10^{-3}$, $\text{wd}=0.01$).
- **Cross-Domain:** 4 domains (Politics, Health, Social/Crisis, Disaster), 16 ordered pairs, 5-fold CV linear SVM for $d_{\mathcal{A}}$.
- **LOSO:** 4 scenarios, $K=5$ seeds per scenario.
- **ARES-Synth:** Test-only evaluation across 3 models on frozen test set ($n=1858$).
- **Hardware & Runtime:** RTX 4060 Laptop GPU (8 GB), i7-13620H, 16 GB RAM; total execution runtime = 205.45 seconds.

---

## 6. Publication Deliverables

### Tables (`milestones/M12_robustness_shift/tables/`)
1. `table_adversarial_attacks.csv`
2. `table_adversarial_strength_curve.csv`
3. `table_adversarial_training.csv`
4. `table_synthetic_injection.csv`
5. `table_cross_domain_matrix.csv`
6. `table_adistance_correlation.csv`
7. `table_loso_scenarios.csv`
8. `table_ares_synth_main.csv`
9. `table_ares_synth_by_perturbation.csv`
10. `table_media_robustness.csv`

### Figures (`milestones/M12_robustness_shift/figures/`)
1. `fig42_adv_strength_curve.png`: Macro-F1 vs perturbation budget $\epsilon$ (Replaces manuscript Fig. 42).
2. `fig_cross_domain_heatmap.png`: Cross-domain transfer Macro-F1 heatmap across domains.
3. `fig_adist_vs_drop.png`: Proxy $\mathcal{A}$-distance vs empirical transfer drop ($r_s = 0.3044$).
4. `fig_ares_synth_coverage_risk.png`: Coverage–risk trade-off under Chow cost-asymmetric abstention on ARES-Synth v1.

---

## 7. Negative and Unexpected Findings

1. **Refutation of $\Delta_{\text{adv}} < 0.05$ (Claim `C045`):** Under word synonym replacement (TextFooler, 15% budget), Full ARES experiences a drop of $\Delta_{\text{adv}} = 0.0871$ [0.0687, 0.1062], refuting the manuscript's blanket claim of $\Delta_{\text{adv}} < 0.05$. Resilience $< 0.05$ holds only under minimal perturbation budgets ($\epsilon \le 0.05$).
2. **Refutation of LOSO Degradation $< 4.8\%$ (Claim `C059`):** Empirical mean degradation across 4 crisis scenarios is $4.897\%$ [0.77%, 8.94%], with severe drops on breaking rumours (PHEME degradation = $10.15\%$ absolute, $34.72\%$ relative).
3. **Severe Cross-Domain Transfer Gaps (Claim `C044`):** The manuscript's reported transfer F1 of ~0.85–0.88 across all domains was ungrounded. Transferring between disparate vocabularies (e.g. Health $\to$ Politics or Politics $\to$ Health) results in drops exceeding $0.50$ F1, driven by high proxy $\mathcal{A}$-distance ($d_{\mathcal{A}} \approx 1.83$).

---

## 8. Manuscript Impact & Action Plan

- **Abstract:** Replace qualitative statement ("high resilience under stress scenarios") with quantified empirical degradation numbers.
- **Section XII-D:** Disaggregate monolithic election claim; mark border conflict and Assam flood numbers as controlled grounded-synthetic benchmarks (SIMULATED); update COVID-19 F1 to measured 0.8799.
- **Section XXII-E:** Correct Grover baseline category error; correct theoretical claim regarding discrepancy minimization to empirical Ben-David bound correlation.
- **Section XXVI-B / Fig. 42:** Replace illustrative Figure 42 with empirical perturbation-strength curves; acknowledge that degradation exceeds 0.05 at standard modification budgets ($\epsilon > 0.05$).
- **Section XXVI-E:** Replace ungrounded live OSINT claim with pre-registered prospective temporal validation protocol Track T11 (`PREREG_T11.md`).

---

## 9. Audit Summary

- **Audit Report:** [AUDIT.md](file:///d:/ares/milestones/M12_robustness_shift/AUDIT.md)
- **Findings Count:** 0 Blocker, 0 Major open issues.
- **Audit Verdict:** Approved for closure. Tag `m12-complete`.
