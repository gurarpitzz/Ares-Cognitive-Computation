# M01 — Corrected Formal Foundations (Theory)

**Status:** COMPLETE  
**Git tag:** `m01-complete`  
**PREREG commit:** `e2c9b18`  

---

## 1. Summary
Milestone M01 establishes the corrected mathematical foundations of ARES, completely auditing and replacing the flawed theoretical claims of the manuscript. We proved that cosine stability under Gaussian perturbation follows an exact noncentral-$t$ distribution (Theorem A, verified across 25 grid cells with KS $p > 0.01$), replacing the manuscript's broken formula $1 - \operatorname{Tr}(\Sigma)/(2r^2)$ which predicted negative similarities. We formulated an exact two-sided UMP invariant mutation test (Theorem B), derived Chow's optimal decision boundaries with calibrated multi-source fusion (Theorem C), and proved the transformer FLOP crossover occurs at $n^* = 6d = 4{,}608$ tokens rather than dominating for $n < d$.

---

## 2. Ledger Claims Addressed

| Ledger ID | Manuscript Claim | Manuscript Value | Measured / Derived Value (95% CI) | New Status | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C070** | Bounded mutation detection guarantee | Exists $\theta$ s.t. $\Delta_t < \theta$ with prob $\ge 1-\delta$ | Exact UMP test via noncentral-$t$; $\alpha=0.05 \implies$ empirical FPR = $0.0501$ ($[0.0488, 0.0514]$) | `CORRECTED-THEORY` | `milestones/M01_theory/results/verify_thmB.json` |
| **C071** | Cosine stability formula | $\mathbb{E}[\cos] \approx 1 - \frac{\operatorname{Tr}(\Sigma)}{2\|x\|^2}$ | Exact $T \sim t'(\nu, r/\sigma)$, expansion $1 - \frac{(d-1)\sigma^2}{2r^2}$, high-d limit $1/\sqrt{1+\kappa}$; KS $p > 0.01$ (25/25) | `CORRECTED-THEORY` | `milestones/M01_theory/results/verify_thmA.json` |
| **C072** | Neyman-Pearson Bayesian verifier | $L(c)$ is Neyman-Pearson optimal | Chow thresholds $\tau_1 = \ln\frac{c_{\text{FP}}-c_a}{c_a}$, $\tau_0 = \ln\frac{c_a}{c_{\text{FN}}-c_a}$; risk $1.2180$; logistic fusion $-13.4\%$ log-loss | `CORRECTED-THEORY` | `milestones/M01_theory/results/verify_thmC.json` |
| **C073** | Ben-David distribution shift bound | ARES explicitly minimizes domain discrepancy | Bound $\hat{d}_{\mathcal{A}} = 2(1-2\cdot\operatorname{err})$; Rademacher bound; ARES bounds risk rather than adversarial minimax | `CORRECTED-THEORY` | `ares_v2/theory/shift.tex` |
| **C075** | Attention dominates for $n < d$ | Layer cost is $\mathcal{O}(n^2 d)$ for typical $n < d$ | $\text{FLOPs} = 24nd^2 + 4n^2d$; crossover $n^* = 6d$ ($4{,}608$ for $d=768$); attention $<10\%$ compute for $n \le 512$ | `CORRECTED-THEORY` | `ares_v2/theory/complexity.tex` |
| **C077** | Linear cosine approximation | $\Delta_t \approx 1 - 0.5\|\epsilon\|^2$ | Expansion valid only for $(d-1)\sigma^2/r^2 \ll 1$; Laurent-Massart concentration $\Delta \xrightarrow{p} 1/\sqrt{1+\kappa}$ | `CORRECTED-THEORY` | `milestones/M01_theory/tables/expansion_error.csv` |
| **C098** | Generalization error scaling | $R_{D_r}(f) \propto 1/|D_r|$ | Standard bounded loss Rademacher minimax rate $\mathcal{O}(1/\sqrt{|D_r|})$; discrete log-likelihood renewal drift model | `CORRECTED-THEORY` | `ares_v2/theory/drift_process.tex` |

---

## 3. What Was Built

| File | Purpose |
| :--- | :--- |
| `ares_v2/theory/notation.tex` | Formal representation setup, coordinate decomposition $a \hat{x} + b$, and spherical geometry. |
| `ares_v2/theory/thmA.tex` | Exact 6-step proof of Theorem A (noncentral-$t$ distribution of cosine similarity). |
| `ares_v2/theory/corollaries.tex` | Small-noise expansion, high-dimensional concentration limit, and anisotropic covariance corollary. |
| `ares_v2/theory/thmB.tex` | Exact two-sided mutation test, Karlin-Rubin MLR proof of UMP optimality, and Laurent-Massart separation. |
| `ares_v2/theory/thmC.tex` | Chow's rule under ternary action space and asymmetric costs; calibrated multi-source fusion. |
| `ares_v2/theory/complexity.tex` | Exact transformer forward FLOP derivation, crossover $n^* = 6d$, and refutation of attention dominance. |
| `ares_v2/theory/shift.tex` | Ben-David et al. (2010) $\mathcal{H}\Delta\mathcal{H}$-divergence and Rademacher generalization error bounds. |
| `ares_v2/theory/drift_process.tex` | Discrete log-likelihood drift and renewal process for temporal claim verification (M06). |
| `ares_v2/theory/ares_theory.tex` | Consolidated master publication-quality LaTeX monograph. |
| `milestones/M01_theory/THEORY_MONOGRAPH.md` | Comprehensive standalone markdown theory monograph with embedded proofs, tables, and figures. |
| `ares_v2/theory/verify/verify_thmA.py` | Numerical verification of Theorem A on a $5 \times 5$ parameter grid ($5 \times 10^6$ Monte Carlo samples). |
| `ares_v2/theory/verify/verify_thmB.py` | Verification of Theorem B false positive rate coverage, theoretical vs empirical ROC, and power curves. |
| `ares_v2/theory/verify/verify_thmC.py` | Verification of Theorem C Chow risk surface optimality and logistic regression source fusion. |
| `ares_v2/theory/verify/verify_anisotropic.py` | Verification of Corollary 3 anisotropic covariance with power-law spectra and parallel energy subtraction. |
| `milestones/M01_theory/ledger_delta.csv` | Pre-registered ledger delta updating 7 theoretical claims to `CORRECTED-THEORY`. |

---

## 4. Data
- **Datasets:** N/A (Theoretical milestone).
- **Simulated Monte Carlo Data:**
  - Synthetic Gaussian perturbations generated via deterministic seeds (`seed=42`).
  - Isotropic and anisotropic noise models across dimensions $d \in \{16, 64, 384, 768, 1024\}$.
  - Total synthetic vectors evaluated: $> 1.2 \times 10^7$ high-dimensional vectors on CUDA.

---

## 5. Experiments

| Experiment ID | Script | Config | Seed | Hardware | Wall-clock (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `M01-verify-thmA-distribution` | `verify_thmA.py` | Grid: $d \in \{16, 64, 384, 768, 1024\}$, $r/s \in \{2, 5, 10, 25, 50\}$ | 42 | RTX 4060 Laptop (CUDA) | 3.54 s |
| `M01-verify-thmB-detection-guarantee` | `verify_thmB.py` | $d=384, s_0=0.04, s_1=0.08, \alpha \in \{0.01, 0.05, 0.10\}$ | 42 | RTX 4060 Laptop (CUDA) | 3.86 s |
| `M01-verify-thmC-decision-fusion` | `verify_thmC.py` | $c_{\text{FP}}=10, c_{\text{FN}}=6, c_a=2, K=4, \rho=0.65$ | 42 | RTX 4060 Laptop (CUDA) | 4.12 s |
| `M01-verify-anisotropic-covariance` | `verify_anisotropic.py` | $d \in \{64, 384, 768\}, \gamma \in \{0.8, 1.5\}, r/s \in \{5, 10, 25\}$ | 42 | RTX 4060 Laptop (CUDA) | 5.21 s |

---

## 6. Results

### Numerical Verification Tables
1. **Theorem A Kolmogorov-Smirnov Test:** `milestones/M01_theory/tables/expansion_error.csv`
   - KS test null hypothesis (identical distributions) was not rejected in any of the 25 cells ($p$-values between $0.024$ and $0.985$, all $> 0.01$).
2. **Anisotropic Covariance Comparison:** `milestones/M01_theory/tables/anisotropic_verification.csv`
   - Corrected expansion error vs manuscript formula: in dominant orientation ($d=384, \gamma=1.5, r/s=25$), corrected error was $0.0365$ vs manuscript error $0.0859$ (manuscript error $> 2.35\times$ higher due to omitting parallel subtraction).

### Figures
- `milestones/M01_theory/figures/thmB_roc_theoretical.png`: Theoretical vs empirical ROC curve matching exactly across $10^5$ samples.
- `milestones/M01_theory/figures/thmB_power_cos_vs_chi2.png`: Power comparison demonstrating the projective information loss of the cosine statistic compared to the full $\chi_d^2$ Euclidean metric.
- `milestones/M01_theory/figures/thmC_bayes_risk_surface.png`: Contour of empirical Bayes risk verifying Chow's theoretical threshold point $(\tau_0 = -0.69, \tau_1 = 1.39)$ as optimal.
- `milestones/M01_theory/figures/thmC_correlated_sources.png`: Posterior calibration comparing unweighted sum vs learned logistic weights ($13.4\%$ log-loss reduction).
- `milestones/M01_theory/figures/anisotropic_spectrum.png`: Cosine stability across dominant, uniform, and minor eigenspaces under power-law spectra.

---

## 7. Deviations from Plan / PREREG
None. All 5 pre-registered hypotheses in `PREREG.md` were derived, proved from first principles, and verified numerically.

---

## 8. Negative or Unexpected Results
- **Taylor Expansion Breakdown:** The small-noise Taylor expansion ($\mathbb{E}[\Delta] \approx 1 - \kappa/2$) breaks down precipitously once the variance ratio $\kappa = (d-1)\sigma^2/r^2 \ge 0.5$. In typical embedding spaces ($d=384$ or $768$), even modest perturbations ($s \ge 0.06$) cause $\kappa > 1$, making Taylor expansions report negative values. The concentration limit $\Delta \xrightarrow{p} 1/\sqrt{1+\kappa}$ must be used instead.
- **Projective Information Loss:** The invariant cosine test loses exactly 1 degree of freedom relative to the full Euclidean test ($\nu = d - 1$ vs $d$). For small perturbations, the $\chi_d^2$ test on $\|\epsilon\|^2$ has strictly higher power than the cosine test, showing that representation norm contains independent corruption information that cosine similarity discards.

---

## 9. Limitations & Threats to Validity
- **Gaussian Noise Assumption:** Theorems A and B assume additive Gaussian perturbations in latent embedding space. While central limit theorems suggest multi-token embedding shifts aggregate approximately normally, heavy-tailed semantic corruptions may exhibit kurtosis that deviates from Student's $t$.
- **Finite Sample Size in Noncentral-$t$ CDF:** Scipy's `nct.cdf` uses AS 243 asymptotic series expansions, which can lose floating-point precision for noncentrality parameter $\lambda > 100$. For extreme low-noise regimes ($r/\sigma > 100$), numerical simulation or saddlepoint approximations are preferred.

---

## 10. Manuscript Impact
The following manuscript sections must be rewritten in the revised paper:
1. **Section XVII-E:** Delete equation $1 - \operatorname{Tr}(\Sigma)/(2r^2)$. Insert Theorem A ($T / \sqrt{T^2 + \nu}$) and Corollary 1 ($1 - (d-1)\sigma^2/(2r^2)$).
2. **Section XVIII:** Delete Proposition 1. Replace with Theorem B (UMP invariant two-sided test with exact false positive rates).
3. **Section XXI:** Delete heuristic decision thresholds. Insert Theorem C (Chow's optimal thresholds with explicit costs and logistic multi-source calibration).
4. **Section XX:** Delete claim that "quadratic attention dominates for $n < d$". Insert correct crossover derivation $n^* = 6d = 4{,}608$.
5. **Section XXVI & XXXV:** Clarify Ben-David $\mathcal{A}$-distance bounds and correct generalization scaling to $\mathcal{O}(1/\sqrt{|D_r|})$.

---

## 11. Reproduce
All experiments can be reproduced on the RTX 4060 Laptop (or CPU) in under 30 seconds:
```bash
# Theorem A
python ares_v2/theory/verify/verify_thmA.py

# Theorem B
python ares_v2/theory/verify/verify_thmB.py

# Theorem C
python ares_v2/theory/verify/verify_thmC.py

# Corollary 3 (Anisotropic)
python ares_v2/theory/verify/verify_anisotropic.py
```

---

## 12. Audit Summary
- **Audit File:** [milestones/M01_theory/AUDIT.md](file:///d:/ares/milestones/M01_theory/AUDIT.md)
- **Findings Count:** 0 BLOCKER, 0 MAJOR, 0 MINOR.
- **Verdict:** **CLOSE**. All theoretical statements are mathematically sound, fully proven, and confirmed by large-sample Monte Carlo verification.
