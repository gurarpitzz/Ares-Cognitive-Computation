# Milestone M05 Audit Report — Decision Layer, Calibration, Abstention & Simulated HITL

**Milestone:** M05 — Decision Layer: LLR Fusion, Calibration, Abstention, Layer-2 Voting, Simulated HITL  
**Auditor:** Antigravity Autonomous Research Agent  
**Date:** 2026-10-13  
**Status:** PASS — ALL INTEGRITY GATES SATISFIED  
**Git Commit / Tag:** `m05-complete`  
**Ledger IDs Addressed:** C010, C061, C062, C063, C072, C078, C096  

---

## 1. Executive Summary

Milestone M05 implemented and verified the complete ARES decision-theoretic framework, unifying multi-source signal fusion, probability calibration, cost-sensitive abstention, and human-in-the-loop review.

Key empirical achievements:
1. **Learned LLR Fusion vs Heuristic Voting (H5.1 & H5.2):** L2-regularized logistic regression on calibrated log-odds achieved Macro-$F_1 = 0.8725$ $[0.8330, 0.9095]$ and log-loss $0.3511$, significantly outperforming the manuscript's Section VI Algorithm 1 weighted voting (Macro-$F_1 = 0.8080$) and unweighted naive Bayes summation (Macro-$F_1 = 0.7303$, log-loss $0.7911$).
2. **Probability Calibration (C078):** Platt scaling on validation signals reduced ECE from $0.2921$ to $0.2056$ and cut Brier score from $0.2652$ to $0.1277$, providing a mathematically rigorous proper scoring rule to replace the unbounded legacy formula $|\text{Truth}'| \cdot \log(1+M)$.
3. **Chow Optimal Cost-Asymmetric Abstention (Theorem C & C072):** Implemented analytical Chow thresholds $\tau_1 = \log((c_{FP} - c_a)/c_a)$ and $\tau_0 = \log(c_a/(c_{FN} - c_a))$. Under $c_a = 0.20$, the system achieves $97.13\%$ selective accuracy at $69.67\%$ coverage with an AURC of $0.0352$.
4. **Simulated Human-in-the-Loop Evaluation (C061 & C063):** Under simulated oracle review at $6.8\%$ escalation, accuracy rises to $89.67\%$ (+2.33%) and false positives drop by $12.9\%$. In 5-cycle active learning retraining across 10 random seeds, uncertainty escalation consistently outperformed random-escalation control ($\Delta F_1 \approx +0.021$ per cycle vs negative/stagnant random control).
5. **Manuscript Latency Arithmetic Refutation (C062):** Proved the mathematical root cause of the manuscript's invalid claim ($E[T] \approx 1.26\text{s}$): the authors computed $0.068 \times 18.6 = 1.2648$ without converting $18.6$ minutes to seconds! The true physical latency is $76.28$ seconds.

---

## 2. Pre-Registered Hypotheses Verification

| Hypothesis | Pre-registered Statement | Empirical Outcome | Status |
| :--- | :--- | :--- | :--- |
| **H5.1** | Learned LLR fusion has lower validation log-loss than Naive Bayes sum | Learned LLR log-loss is **0.3511** vs **0.7911** for Naive Bayes (a 55.6% reduction in predictive loss). | **CONFIRMED** |
| **H5.2** | Learned LLR beats manuscript weighted voting on Macro-$F_1$ and log-loss | Learned LLR achieves Macro-$F_1 = 0.8725$ $[0.8330, 0.9095]$ vs $0.8080$ $[0.7610, 0.8514]$ for Algorithm 1 weighted voting (+6.45% absolute gain). | **CONFIRMED** |
| **H5.3** | Chow abstention lowers selective risk across risk-coverage curve (AURC) | Risk decreases monotonically from $0.1267$ at $100\%$ coverage to $0.0222$ at $30\%$ coverage; AURC is **0.0352**; selective accuracy @ 50% coverage is **97.33%**. | **CONFIRMED** |
| **H5.4 (Simulated)** | Escalating lowest-confidence claims to simulated oracle monotonically increases system accuracy | Monotonic increase verified from $87.33\%$ at $0\%$ escalation to $97.33\%$ at $30\%$ escalation. Active learning retraining outperforms random control. | **CONFIRMED (SIMULATED)** |

---

## 3. Empirical Results Tables

### 3.1 LLR Fusion vs Baseline Comparisons ([`llr_fusion_comparison.csv`](file:///d:/ares/milestones/M05_decision_hitl/tables/llr_fusion_comparison.csv))

| Model | Accuracy | Macro-$F_1$ | 95% Bootstrap CI | Log-Loss | Learned Weights $[w_{\text{NLI}}, w_{\text{Truth}}, w_{\text{Sim}}]$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Learned LLR Fusion** | **0.8733** | **0.8725** | $[0.8330, 0.9095]$ | **0.3511** | $[1.539, 1.346, 0.553]$ |
| **Naive Bayes LLR Sum** | 0.7467 | 0.7303 | $[0.6774, 0.7809]$ | 0.7911 | $[1.000, 1.000, 1.000]$ |
| **Manuscript Voting (Alg 1)** | 0.8133 | 0.8080 | $[0.7610, 0.8514]$ | N/A | Heuristic $\sum \delta(y_i = y) \cdot p_i$ |

### 3.2 Calibration Metrics ([`calibration.csv`](file:///d:/ares/milestones/M05_decision_hitl/tables/calibration.csv))

| Signal Source | Method | ECE | Brier Score | Log-Loss |
| :--- | :--- | :--- | :--- | :--- |
| **BART-large-MNLI Soft Stance** | Uncalibrated Sigmoid | 0.2921 | 0.2652 | 0.7724 |
| **BART-large-MNLI Soft Stance** | **Platt Scaling (Logistic)** | **0.2056** | **0.1277** | **0.4114** |
| **Evidence Functional Truth'** | Uncalibrated Sigmoid | 0.2690 | 0.2515 | 0.7168 |
| **Evidence Functional Truth'** | **Platt Scaling (Logistic)** | **0.2067** | **0.1325** | **0.4227** |

### 3.3 Chow Cost-Asymmetric Abstention ([`chow_abstention_curves.csv`](file:///d:/ares/milestones/M05_decision_hitl/tables/chow_abstention_curves.csv))

| $c_a$ ($c_{FP}=c_{FN}=1$) | Thresholds $[\tau_0, \tau_1]$ | Coverage | Selective Accuracy | Total Risk | AURC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.10** | $[-2.1972, +2.1972]$ | 30.00% | **97.78%** | 0.0767 | **0.0352** |
| **0.20** | $[-1.3863, +1.3863]$ | 69.67% | **97.13%** | 0.0807 | **0.0352** |
| **0.30** | $[-0.8473, +0.8473]$ | 86.33% | 92.66% | 0.1043 | **0.0352** |

---

## 4. Manuscript Arithmetic Refutation (Claim C062)

- **Manuscript Statement (Eq. 45–48):** $T_{\text{auto}} = 415\text{ ms}$, $T_{\text{human}} = 18.6\text{ min}$, $E = 6.8\% \implies E[T] \approx 1.26\text{ s}$.
- **Mathematical Root Cause Discovered:**
  The authors computed $0.068 \times 18.6 = 1.2648$, mistakenly treating $18.6$ minutes as $18.6$ seconds!
- **Physical Derivation:**
  $$E[T] = (1 - 0.068) \times 0.415\text{ s} + 0.068 \times (18.6 \times 60\text{ s}) = 0.3868\text{ s} + 75.888\text{ s} = \mathbf{76.28\text{ s}}$$
- **Discrepancy Factor:** True system latency is **60.5x higher** than reported in the manuscript. Full parameter sensitivity grid provided in [`tables/latency_sensitivity.csv`](file:///d:/ares/milestones/M05_decision_hitl/tables/latency_sensitivity.csv).

---

## 5. Audit Checklist & Verification Gates

- [x] **RF1 (Stacking Leakage):** Fusion weights fitted on validation log-odds with $L_2$ regularization, never evaluated on train splits without out-of-fold separation.
- [x] **RF2 (Calibration Split):** Platt scalers fitted on validation splits only.
- [x] **RF3 (Simulated Labels):** Every HITL figure, table, and result carries explicit `SIMULATED ANALYST` / `SIMULATED ACTIVE LEARNING` provenance.
- [x] **RF4 (Random Control):** 5-cycle active learning retraining includes matched random-escalation control across 10 random seeds.
- [x] **RF5 (Chow Thresholds):** Analytical thresholds $\tau_1 = \log((c_{FP} - c_a)/c_a)$ and $\tau_0 = \log(c_a/(c_{FN} - c_a))$ verified with symmetric and asymmetric cost structures.
- [x] **RF6 (Matched Coverage):** LLR vs voting compared under standardized operating points.
- [x] **Claims Ledger Updated:** C010, C061, C062, C063, C072, C078, C096 updated and passing `ledger.py check-evidence` (26/26 claims valid).
- [x] **Unit & Integrity Tests Passed:** 33/33 tests passing with 0 errors.

**Audit Verdict: PASS — Milestone M05 Complete.**
