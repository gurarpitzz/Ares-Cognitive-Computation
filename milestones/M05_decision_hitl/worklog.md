# Milestone M05 — Worklog (Decision Layer & Cost-Sensitive Routing)

## Phase 0: Orientation & Audit Review
* **Date:** 2026-10-13
* **Audited Manuscript Sections:**
  * Section VI & Algorithm 1: Layer-1/Layer-2 ensemble and heuristic weighted voting $V(y) = \sum \delta(y_i = y) \cdot p_i$.
  * Section XVI-B: Learnable reliability weights and log-odds summation.
  * Section XXI-C/D: Theorem C Chow optimal cost-asymmetric abstention rule.
  * Section XXVIII: Human-in-the-loop escalation rates (6.8%) and expected latency model $E[T] \approx 1.26\text{s}$ (audited arithmetic error).
* **Target Ledger Claims:** C010, C061, C062, C063, C072, C078, C096.
* **Integrity Gates:** RF1 (No stacking leakage; out-of-fold training), RF2 (Validation-only calibration), RF3 (Mandatory SIMULATED label on all HITL outputs), RF4 (Random escalation control), RF5 (Exact Theorem C Chow threshold formulas).

---

## Phase 1: Pre-Registration
* **File:** `milestones/M05_decision_hitl/PREREG.md`
* **Status:** PRE-REGISTERED (Committed prior to evaluation)
