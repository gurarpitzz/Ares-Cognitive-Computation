# M05 — Decision Layer: LLR Fusion, Calibration, Abstention, Layer-2 Voting, Simulated HITL

**Objective:** Implement the decision-theoretic verifier of Sections XVI/XXI (Theorem C from M01) and the triple-layer engine of Section VI. Measure calibration, abstention trade-offs, and human-in-the-loop effects, using **simulated** analysts and labelling them as simulated everywhere.
**Ledger IDs:** C010, C061, C062, C063, C072, C078, C096.
**Depends on:** M01, M03, M04.
**Compute:** RTX 4060 Laptop. Mostly CPU on the cached L1 outputs; about 5 GPU-h for the HITL retraining cycles (DeBERTa-v3-base, resumable).
**Output dir:** `milestones/M05_decision_hitl/`

---

## Pre-registration
- **H5.1:** Learned-weight LLR fusion (logistic regression on calibrated per-source log-odds) has lower validation log-loss than the naive-Bayes sum (w = 1) when sources are correlated.
- **H5.2:** LLR fusion beats the manuscript's weighted voting V(y) = Σ δ(yᵢ = y)·pᵢ on macro-F1 and log-loss (T1/T2 datasets).
- **H5.3:** Chow abstention lowers selective risk; report the full risk–coverage curve (AURC) and the operating points for pre-registered costs c_a ∈ {0.1, 0.2, 0.3} (c_FP = c_FN = 1) **and** the paper's τ = 0.65.
- **H5.4 (simulated):** escalating the lowest-confidence fraction to an oracle analyst increases system accuracy monotonically in escalation rate. Report the curve. No real-world claim.

## Steps

### M05.1 — Assemble per-source signals
- **Action:** Load cached out-of-fold (train) and standard (val/test) outputs: DeBERTa-v3-base, LLM zero-shot, JBC (T3), evidence features (Truth' hard/soft, weighted entail/contradict counts, max S), logic features. `src/ares/decision/signals.py` builds a per-dataset feature table keyed by item_id.
- **Done when:** tables built; no item lacks out-of-fold train signals.

### M05.2 — Per-source calibration
- **Action:** For each source, fit temperature scaling / Platt / isotonic on **validation**, choose by val log-loss via nested CV (or a split of val), convert to calibrated log-odds ℓᵢ. Report ECE/Brier before/after; reliability diagrams.
- **Output:** `figures/reliability_<source>_<dataset>.pdf`, `tables/calibration.csv`.

### M05.3 — LLR fusion
- **Action:** `decision/llr.py`: (a) naive-Bayes sum L = log-prior + Σ ℓᵢ; (b) learned weights via L2 logistic regression on ℓᵢ, trained on **out-of-fold train** signals, with regularisation chosen on val. Report learned wᵢ with bootstrap CIs (these are the "learnable reliability weights" of Sec. XVI-B) and pairwise source correlations.
- **Done when:** H5.1 evaluated.

### M05.4 — Manuscript Layer-1/Layer-2 engine (Sec. VI, Alg. 1)
- **Action:** `decision/voting.py`: implement Alg. 1 exactly: each model gives (yᵢ, pᵢ); V(y) = Σ δ(yᵢ = y)·pᵢ; ŷ = argmax; escalate if max V < τ (τ ∈ [0.5, 0.7]; declared 0.65, C069). Compare with LLR fusion (H5.2) at matched escalation rates.

### M05.5 — Three-way decisions & abstention
- **Action:** `decision/abstain.py`: Chow thresholds τ1 = log((c_FP − c_a)/c_a), τ0 = log(c_a/(c_FN − c_a)) applied to calibrated posterior log-odds. Datasets with a native NEI class (FEVER, AVeriTeC): report both (i) NEI as a predicted class and (ii) NEI-as-abstention, and define how each is scored in PREREG. Risk–coverage curves, AURC, selective accuracy @ coverage {50, 70, 80, 90, 100 %}.
- **This resolves audit I-11 / I-12:** the unified decision rule in the paper becomes "calibrated posterior + Chow thresholds". All four old rules are retired.

### M05.6 — Escalation rule (C096)
- **Action:** Escalate if (confidence below the τ rule) OR (ρ(c) > ρ_crit = 0.75, declared). ρ comes from M10, so use a placeholder ρ = 0 here; M11 reruns with the real ρ. Report how many escalations each condition triggers.

### M05.7 — Simulated HITL (C061, C063)
- **Action:** `decision/hitl_sim.py`, `data_kind: "simulated"`:
  1. **Static:** analyst = gold-label oracle; also noisy analysts with accuracy a ∈ {0.80, 0.90, 0.95}. Sweep escalation rate 0–30 %. Report system accuracy/macro-F1, false-positive reduction, and escalation rate by domain (domains from dataset source/topic per M10's pre-registered mapping).
  2. **Retraining cycles:** time-ordered stream where dates exist (AVeriTeC, X-FACT), else a seeded random order. Pool → model → escalate lowest-confidence k % → add analyst labels → retrain the fusion head (and optionally fine-tune DeBERTa) → repeat 5 cycles × 10 seeds. Report ΔF1 per cycle and escalation trend with CIs, against a **random-escalation control** (otherwise gains might just come from more labels).
- **Output:** `figures/hitl_*.pdf`, `tables/hitl_*.csv`, each captioned "SIMULATED analyst".

### M05.8 — Expected latency model (C062)
- **Action:** E[T] = (1 − E)·T_auto + E·T_human. T_auto comes from M13 (placeholder until then). T_human is **not measured**: show a sensitivity plot for T_human ∈ {1, 5, 18.6, 60} min and E ∈ [0, 0.2]. Explicitly show that the manuscript's own values give ≈ 76 s, not 1.26 s.

### M05.9 — Calibrated confidence output
- **Action:** The system's "confidence" field = calibrated P(predicted label). Replaces Conf_POC (C078) in the service (M14).

---

## Audit gate — milestone-specific red flags
- RF1: **Stacking leakage:** fusion trained on in-fold L1 outputs would inflate results. The auditor verifies out-of-fold provenance for a random sample of items.
- RF2: Calibration fitted on validation only, never test.
- RF3: Every HITL table/figure/JSON carries "SIMULATED"; nothing implies real analysts.
- RF4: Random-escalation control present for retraining cycles.
- RF5: Chow-threshold implementation matches M01 Theorem C (auditor checks the signs with a toy example).
- RF6: The comparison of voting vs LLR is at matched coverage (otherwise it's unfair).

## Exit criteria
- [ ] Calibration, LLR fusion, voting, and abstention implemented and evaluated on T1/T2 datasets.
- [ ] Simulated HITL static + cycles with controls.
- [ ] Ledger: C010, C072, C078, C096 updated. C061, C063 → REVISED (simulated) or REMOVE per results. C062 → arithmetic corrected; T_human marked not measured.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m05-complete`.
