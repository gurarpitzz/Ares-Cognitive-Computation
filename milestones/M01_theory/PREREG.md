# M01 Pre-Registration — Corrected Formal Foundations (Theory)

**Milestone:** M01 — Corrected Formal Foundations (Theory)  
**Date:** 2026-10-13  
**Status:** PRE-REGISTERED  
**Target Ledger Claims:** C070, C071, C072, C073, C075, C077, C098 (+ C078)  
**Primary Verification Metrics:** Kolmogorov-Smirnov $p$-value, Empirical FPR coverage, Bayes risk gap, FLOP ratio  
**Compute:** RTX 4060 Laptop (Intel Core i7-13620H 16 threads + CUDA for fast MC tensor sampling)  

---

## 1. Scope & Objective
Replace the mathematically flawed statements of Sections XVI–XXI with rigorous theorems, complete proofs, and large-sample Monte Carlo verification. Every formula that replaces a manuscript claim must be proven from first principles and verified numerically to at least 3 significant digits across dimensional grids.

---

## 2. Pre-Registered Hypotheses & Tests

### 2.1 Theorem A: Exact Distribution of Cosine Stability
* **Setting:** $x \in \mathbb{R}^d$ fixed with $r = \|x\| > 0$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I_d)$, $d \ge 2$, degrees of freedom $\nu = d - 1$.
* **Hypothesis 1:** $\Delta = \cos(x, x+\epsilon) = T / \sqrt{T^2 + \nu}$ where $T \sim \text{noncentral-}t(\nu, \lambda = r/\sigma)$.
* **Test Check:** Two-sample Kolmogorov-Smirnov test comparing empirical Monte Carlo distribution ($N = 2 \times 10^5$ samples per cell) against scipy's analytical noncentral-$t$ CDF `scipy.stats.nct.cdf`.
  * Grid: $d \in \{16, 64, 384, 768, 1024\}$, $r/\sigma \in \{2, 5, 10, 25, 50\}$.
  * Criterion: Two-sided KS test $p$-value $> 0.01$ across all 25 cells (null hypothesis of identical distributions not rejected).

### 2.2 Corollaries: Small-Noise & High-Dimensional Asymptotics
* **Hypothesis 2 (Expansion vs Manuscript):** 
  * The second-order expansion is $\mathbb{E}[\Delta] = 1 - \frac{(d-1)\sigma^2}{2r^2} + O(\sigma^4/r^4)$.
  * The manuscript's formula $1 - \frac{\text{Tr}(\Sigma)}{2r^2} = 1 - \frac{d\sigma^2}{2r^2}$ is inaccurate in low dimensions and catastrophically underestimates $\mathbb{E}[\Delta]$ when $(d-1)\sigma^2/r^2 \ge 0.5$, predicting negative or near-zero values when the true mean is $\approx 1/\sqrt{1 + (d-1)\sigma^2/r^2}$.
* **Test Check:** Tabulate exact numerical integration vs manuscript formula vs corrected expansion across $(d-1)\sigma^2/r^2 \in \{0.01, 0.1, 0.5, 1.0, 2.0\}$.

### 2.3 Theorem B: Two-Sided Mutation Detection Guarantee
* **Setting:** $H_0: \epsilon \sim \mathcal{N}(0, \sigma_0^2 I)$ (benign paraphrase) vs $H_1: \epsilon \sim \mathcal{N}(0, \sigma_1^2 I)$ with $\sigma_1 > \sigma_0$.
* **Hypothesis 3:**
  1. The threshold test $\Delta < \theta_\alpha$ with $\theta_\alpha = g(F_{t'(\nu, r/\sigma_0)}^{-1}(\alpha))$ achieves exact type-I error rate $\alpha$.
  2. The empirical false positive rate across $10^5$ samples falls within the $95\%$ Wilson binomial confidence interval of $\alpha$.
  3. The family of noncentral-$t$ distributions has Monotone Likelihood Ratio (MLR) in $T$, rendering the threshold test Uniformly Most Powerful (UMP) among all cosine-based invariant tests (Karlin-Rubin Theorem).
  4. Finite-sample guarantee: Using Laurent-Massart $\chi^2$ bounds, provide an explicit separation $\sigma_1^2 - \sigma_0^2$ guaranteeing simultaneous FPR $\le \delta_0$ and FNR $\le \delta_1$.

### 2.4 Theorem C: Decision Layer & Chow Abstention
* **Hypothesis 4:**
  1. Given misclassification costs $c_{FP}, c_{FN}$ and abstention cost $c_a$, expected loss is strictly minimized by Chow's thresholding rule:
     $$\tau_1 = \ln\frac{c_{FP}-c_a}{c_a}, \quad \tau_0 = \ln\frac{c_a}{c_{FN}-c_a}$$
     with non-empty abstention region if and only if $c_a < \frac{c_{FP}c_{FN}}{c_{FP}+c_{FN}}$.
  2. Under correlated evidence sources, logistic regression on calibrated source log-odds achieves strictly lower log-loss than the unweighted sum ($w_i = 1$).

### 2.5 Transformer Complexity Crossover (C075)
* **Hypothesis 5:**
  * Transformer layer forward FLOPs $= 24nd^2 + 4n^2d$.
  * Attention quadratic cost equals FFN/projection cost at sequence length $n^* = 6d$ ($4,608$ tokens for $d=768$).
  * For standard sequence lengths $n \le 512$, the attention term contributes $< 10\%$ of FLOPs, disproving the claim that "quadratic attention dominates for $n < d$".

---

## 3. Decision Rules
1. M01 cannot close unless all mathematical proofs are complete and peer-review ready.
2. M01 cannot close unless all numerical KS tests in `verify_thmA.py` pass with $p > 0.01$.
3. All result JSON files must carry `"data_kind": "simulated"`.
4. Claims C070, C071, C072, C073, C075, C077, C098 are updated to `CORRECTED-THEORY` in `claims_ledger.csv`.
