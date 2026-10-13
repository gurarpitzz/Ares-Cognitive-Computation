# Milestone M01 — Worklog (Theory & Formal Foundations)

## Phase 0: Orientation & Audit Review
* **Date:** 2026-10-13
* **Audited Manuscript Sections:**
  * Section XVI: Veracity likelihood formulation and Neyman-Pearson claim.
  * Section XVII: Mutation model $\Delta_t$, expected cosine formula $1 - \text{Tr}(\Sigma)/(2r^2)$, claim of "immunity".
  * Section XVIII: Proposition 1 (vacuous $1-\delta$ detection guarantee without type-I error or Lipschitz bound).
  * Section XX: Complexity and FLOP scaling ("attention dominates for $n < d$").
  * Section XXI: Decision thresholds $\tau_0, \tau_1$ and $\Delta_t \approx 1 - 0.5\|\epsilon\|^2$.
  * Section XXVI: Ben-David distribution shift bound ("ARES explicitly minimizes discrepancy").
  * Section XXXV: Generalization error bound $R \propto 1/|D_r|$.
* **Ledger Target Claims:** C070, C071, C072, C073, C075, C077, C098 (+ C078).

---

## Phase 1: Pre-Registration
* **File:** `milestones/M01_theory/PREREG.md` created.
* **Pre-registration Scope:** Exact noncentral-t distribution (Theorem A), two-sided UMP detection guarantee (Theorem B), Chow cost-optimal abstention (Theorem C), complexity crossover derivation $n^* = 6d$, and large-sample Monte Carlo verification.
* **Committed:** commit `e2c9b18`.

---

## Phase 2: Build (Mathematical Derivations & LaTeX Monograph)
* **LaTeX Source Files:**
  * `ares_v2/theory/notation.tex`: Formal geometric coordinate setup and spherical decomposition.
  * `ares_v2/theory/thmA.tex`: Exact 6-step proof of noncentral-t transformation $\Delta = T/\sqrt{T^2+\nu}$.
  * `ares_v2/theory/corollaries.tex`: Small-noise expansion $1 - (d-1)\sigma^2/(2r^2)$, high-d limit $1/\sqrt{1+\kappa}$, anisotropic projection subtraction $\operatorname{Tr}(\Sigma) - \hat{x}^\top\Sigma\hat{x}$.
  * `ares_v2/theory/thmB.tex`: Two-sided UMP mutation detection guarantee via Karlin-Rubin MLR.
  * `ares_v2/theory/thmC.tex`: Chow optimal abstention thresholds $\tau_0, \tau_1$ and multi-source logistic regression.
  * `ares_v2/theory/complexity.tex`: Transformer forward FLOP derivation $24nd^2 + 4n^2d$ and crossover $n^* = 6d$.
  * `ares_v2/theory/shift.tex`: Ben-David et al. (2010) $\mathcal{H}\Delta\mathcal{H}$-divergence and Rademacher generalization bounds.
  * `ares_v2/theory/drift_process.tex`: Discrete log-likelihood drift model for M06.
  * `ares_v2/theory/ares_theory.tex`: Consolidated master LaTeX monograph.
  * `milestones/M01_theory/THEORY_MONOGRAPH.md`: Self-contained markdown monograph with embedded proofs, tables, and figures.

---

## Phase 3: Test
* Ran integrity and test suites:
  * `pytest tests/` passed 18/18 tests.
  * Verified determinism and random seeds across all numerical test scripts.

---

## Phase 4: Run (Numerical Monte Carlo Verification)
1. **Theorem A:**
   * Script: `ares_v2/theory/verify/verify_thmA.py`
   * Executed 25 grid cells ($5 \times 10^6$ samples).
   * Result: 25/25 KS tests passed ($p > 0.01$). Output: `verify_thmA.json`, `expansion_error.csv`.
2. **Theorem B:**
   * Script: `ares_v2/theory/verify/verify_thmB.py`
   * Evaluated $\alpha \in \{0.01, 0.05, 0.10\}$ across $10^5$ samples.
   * Result: All empirical FPRs within 95% Wilson binomial CI. Output: `verify_thmB.json`, `thmB_roc_theoretical.png`, `thmB_power_cos_vs_chi2.png`.
3. **Theorem C:**
   * Script: `ares_v2/theory/verify/verify_thmC.py`
   * Evaluated Chow risk surface on 2D grid and correlated sources fusion ($\rho = 0.65$).
   * Result: Chow thresholds minimize empirical Bayes risk (discrepancy $< 0.016$); logistic regression improves log-loss by $13.44\%$. Output: `verify_thmC.json`, `thmC_bayes_risk_surface.png`, `thmC_correlated_sources.png`.
4. **Corollary 3 (Anisotropic Covariance):**
   * Script: `ares_v2/theory/verify/verify_anisotropic.py`
   * Evaluated power-law decaying spectra ($\gamma \in \{0.8, 1.5\}$) across dimensions $d \in \{64, 384, 768\}$.
   * Result: Corrected expansion reduces error by $> 2.3\times$ compared to manuscript formula by accounting for parallel projection. Output: `verify_anisotropic.json`, `anisotropic_verification.csv`, `anisotropic_spectrum.png`.

---

## Phase 5: Report & Ledger Delta
* Created `milestones/M01_theory/ledger_delta.csv`.
* Applied delta via `python ares_v2/scripts/ledger.py apply-delta M01_theory`.
* Validated ledger integrity and evidence existence via `python ares_v2/scripts/ledger.py check-evidence`.
* Generated `milestones/M01_theory/README.md`.

---

## Phase 6: Adversarial Audit
* Auditor review conducted and documented in `milestones/M01_theory/AUDIT.md`.
* Verdict: CLOSE.

---

## Phase 7--9: Close
* Tagged `m01-complete`.
* Pushed commits and tags to GitHub origin.

