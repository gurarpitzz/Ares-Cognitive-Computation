# Milestone M05 — Decision Layer, Calibration, Chow Abstention & Simulated HITL

## Overview

Milestone M05 implements the decision layer for the ARES architecture, translating multi-source evidence and text verifier signals into calibrated posterior probabilities, decision-theoretic Chow abstention rules (Theorem C), and simulated human-in-the-loop escalation workflows.

## Key Deliverables

1. **Source Code:**
   - [`ares.decision.calibration`](file:///d:/ares/ares_v2/src/ares/decision/calibration.py): Platt logistic scaling, single-parameter temperature scaling, Expected Calibration Error (ECE), and Brier scores.
   - [`ares.decision.llr`](file:///d:/ares/ares_v2/src/ares/decision/llr.py): Unweighted Naive Bayes LLR summation vs Learned $L_2$-regularized Logistic LLR Fusion.
   - [`ares.decision.voting`](file:///d:/ares/ares_v2/src/ares/decision/voting.py): Manuscript Section VI Algorithm 1 Heuristic Weighted Voting engine.
   - [`ares.decision.abstain`](file:///d:/ares/ares_v2/src/ares/decision/abstain.py): Theorem C Chow cost-asymmetric abstention rule, risk-coverage curves, and AURC.
   - [`ares.decision.hitl_sim`](file:///d:/ares/ares_v2/src/ares/decision/hitl_sim.py): Simulated oracle and noisy analyst review ($a \in \{0.80, 0.90, 0.95\}$) and 5-cycle active learning retraining vs random controls.
   - [`ares.decision.latency_model`](file:///d:/ares/ares_v2/src/ares/decision/latency_model.py): Expected end-to-end latency formulation and mathematical proof of the Claim C062 arithmetic flaw.

2. **Empirical Results Tables:**
   - [`tables/llr_fusion_comparison.csv`](file:///d:/ares/milestones/M05_decision_hitl/tables/llr_fusion_comparison.csv): Learned LLR vs Naive Bayes vs Voting.
   - [`tables/calibration.csv`](file:///d:/ares/milestones/M05_decision_hitl/tables/calibration.csv): ECE, Brier score, and log-loss reductions from Platt scaling.
   - [`tables/chow_abstention_curves.csv`](file:///d:/ares/milestones/M05_decision_hitl/tables/chow_abstention_curves.csv): Operating points across $c_a \in \{0.1, 0.2, 0.3\}$, selective accuracy milestones, and AURC.
   - [`tables/hitl_simulation.csv`](file:///d:/ares/milestones/M05_decision_hitl/tables/hitl_simulation.csv): Escalation rate sweeps (0% to 30%) with domain-specific breakdowns.
   - [`tables/hitl_active_learning.csv`](file:///d:/ares/milestones/M05_decision_hitl/tables/hitl_active_learning.csv): 5-cycle active learning retraining vs random control across 10 seeds.
   - [`tables/latency_sensitivity.csv`](file:///d:/ares/milestones/M05_decision_hitl/tables/latency_sensitivity.csv): Sensitivity grid across automation latencies, human latencies, and escalation rates.

3. **Experiment Results:**
   - [`results/M05-decision-layer-s42.json`](file:///d:/ares/milestones/M05_decision_hitl/results/M05-decision-layer-s42.json): Provenance-tracked result artifact conforming to `ExperimentResult` Pydantic schema.

4. **Claims Ledger Delta:**
   - [`ledger_delta.csv`](file:///d:/ares/milestones/M05_decision_hitl/ledger_delta.csv): Updated Claims C010, C061, C062, C063, C072, C078, C096.

## Running Tests & Evaluation

```bash
# Run decision layer unit tests
pytest ares_v2/tests/test_decision.py -v

# Run full evaluation script
python ares_v2/scripts/run_m05_evaluation.py

# Run repository integrity checks
python ares_v2/scripts/ledger.py check-evidence
pytest ares_v2/tests/ -v
```
