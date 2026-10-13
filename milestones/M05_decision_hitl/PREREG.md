# M05 Pre-Registration — Decision Layer: LLR Fusion, Calibration, Chow Abstention, Simulated HITL

**Milestone:** M05 — Decision Layer & Cost-Sensitive Routing  
**Date:** 2026-10-13  
**Status:** PRE-REGISTERED  
**Target Ledger Claims:** C010 (Weighted voting vs LLR), C061 (Escalation rates), C062 (Expected latency model), C063 (HITL cycles), C072 (Theorem C Chow optimality), C078 (Confidence output), C096 (Escalation rule)  
**Primary Metrics:** Macro-F1, Multi-class Brier Score, Log-Loss, Area Under the Risk-Coverage Curve (AURC)  
**Secondary Metrics:** Expected Calibration Error (ECE), Selective Accuracy at Fixed Coverage (50%, 70%, 80%, 90%, 100%)  
**Compute Hardware:** Intel Core i7-13620H (16 threads), NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)  

---

## 1. Scope & Objective

Implement and empirically benchmark the ARES Decision Layer:
1. **Per-Source Calibration:** Fit Platt/temperature scaling on validation splits to convert uncalibrated logits from Layer-1 (text verifiers) and Layer-2 (evidence engine) into proper log-odds $\ell_i$.
2. **Log-Likelihood Ratio (LLR) Fusion:** Compare Naive-Bayes summation against L2-regularized logistic fusion weights (verifying Theorem C from Milestone M01).
3. **Chow Optimal Cost-Asymmetric Abstention:** Apply analytical decision thresholds:
   $$\tau_1 = \log \frac{c_{\text{FP}} - c_a}{c_a}, \quad \tau_0 = \log \frac{c_a}{c_{\text{FN}} - c_a}$$
   and evaluate the full risk-coverage frontier.
4. **Simulated Human-in-the-Loop (HITL):** Simulate escalation triage queues across oracle and noisy analysts ($a \in \{0.80, 0.90, 0.95\}$) with active learning retraining cycles against a random escalation control.

---

## 2. Pre-Registered Hypotheses

### H5.1: Learned LLR Fusion vs Naive Bayes Summation
- **Hypothesis:** Learned-weight LLR fusion (L2 logistic regression over calibrated source log-odds) achieves strictly lower validation log-loss and Brier score than unweighted Naive Bayes summation ($\mathbf{w} = \mathbf{1}$) under correlated text and evidence signals.

### H5.2: LLR Fusion vs Heuristic Weighted Voting (Algorithm 1)
- **Hypothesis:** Decision-theoretic LLR fusion outperforms the manuscript's heuristic weighted voting rule ($V(y) = \sum_i \delta(y_i = y) \cdot p_i$) on Macro-$F_1$ and calibration ECE at matched escalation thresholds.

### H5.3: Chow Abstention Risk Reduction
- **Hypothesis:** Implementing Chow's cost-asymmetric abstention rule strictly lowers empirical selective risk compared to full coverage across pre-registered abstention costs $c_a \in \{0.10, 0.20, 0.30\}$ with $c_{\text{FP}} = c_{\text{FN}} = 1.0$.

### H5.4: Simulated HITL Monotonicity
- **Hypothesis (Simulated):** Escalating the lowest-confidence fraction of predictions to an oracle analyst yields a strictly monotonic increase in effective system accuracy as escalation rate sweeps from $0\%$ to $30\%$.

---

## 3. Integrity Gates & Negative Constraints

1. **RF1 (No Stacking Leakage):** Fusion weights must be trained on strictly out-of-fold predictions. No in-fold training data may be used for meta-model fitting.
2. **RF2 (Validation Calibration):** Temperature scaling and Platt parameters must be fitted exclusively on validation splits, never on the official test set.
3. **RF3 (Simulated HITL Labeling):** Every table, figure, and JSON artifact reporting human-in-the-loop metrics must be explicitly captioned as `SIMULATED ANALYST` to prevent real-world deployment misrepresentation.
4. **RF4 (Active Learning Control):** Retraining cycles must include a random-escalation control to isolate active learning gains from sample-size expansion.
5. **RF5 (Theorem C Fidelity):** Decision thresholds must match the analytical Chow optimality formulas derived in M01 Theorem C.
