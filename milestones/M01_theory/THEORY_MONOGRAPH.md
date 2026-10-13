# Rigorous Mathematical Foundations and Statistical Correction of ARES Cognitive Computation

**Monograph ID:** ARES-THEORY-2026-M01  
**Authors:** ARES Core Engineering & Verification Team  
**Date:** October 2026  
**Status:** FULLY GROUNDED & VERIFIED  

---

## Abstract
This monograph provides the complete mathematical and statistical foundations for the Autonomous Real-time Evaluation System (ARES), resolving foundational mathematical flaws in the original manuscript (*Ares: Cognitive Computation*). Specifically:
1. **Exact Distribution of Cosine Stability (Theorem A):** We prove that under isotropic Gaussian perturbation $\epsilon \sim \mathcal{N}(0, \sigma^2 I_d)$, the cosine similarity statistic $\Delta = \cos(x, x+\epsilon)$ follows a noncentral-$t$ distribution transformation $\Delta = T / \sqrt{T^2 + \nu}$ where $T \sim t'(\nu = d-1, \lambda = \|x\|/\sigma)$. This refutes the manuscript's heuristic $1 - \operatorname{Tr}(\Sigma)/(2r^2)$, which produces impossible negative values for standard noise regimes. The exact distribution is verified across 25 parameter cells ($5 \times 10^6$ Monte Carlo samples) via Kolmogorov-Smirnov tests ($p > 0.01$).
2. **Two-Sided Mutation Detection Guarantee (Theorem B):** We replace the vacuous Proposition 1 with an exact two-sided hypothesis test that is proven to be Uniformly Most Powerful (UMP) among all orthogonal-invariant cosine tests via the Karlin-Rubin Monotone Likelihood Ratio theorem.
3. **Chow's Optimal Decision Verifier (Theorem C):** We model the three-way verifier decision problem (True, False, HITL Abstention) under asymmetric operational costs, deriving exact optimal decision boundaries $\tau_0, \tau_1$ and proving that unweighted source summation is sub-optimal under correlated evidence.
4. **Transformer FLOP Complexity (Hypothesis 5):** We derive the exact forward FLOP scaling $24nd^2 + 4n^2d$, proving that the crossover point where quadratic attention equals linear projection is $n^* = 6d$ ($4{,}608$ tokens for $d=768$). This refutes the manuscript's claim that attention dominates for $n < d$.
5. **Distribution Shift and Generalization Bounds:** We ground domain transfer via Ben-David et al.'s $\mathcal{A}$-distance and Rademacher generalization bounds, correcting the $1/|D_r|$ convergence claim to the standard $1/\sqrt{|D_r|}$ statistical rate.

---

## 1. Notation and Setup

Let $x \in \mathbb{R}^d$ be a fixed, non-zero representation vector with Euclidean norm $r \triangleq \|x\|_2 > 0$ and unit direction $\hat{x} \triangleq x / r$. In representation space, $x$ corresponds to an unperturbed reference embedding (e.g., from DeBERTa-v3 or RoBERTa).

Let $\epsilon \sim \mathcal{N}(0, \Sigma)$ denote additive perturbation noise. Under the isotropic model, $\Sigma = \sigma^2 I_d$. Under anisotropic mutation, $\Sigma = \operatorname{diag}(\lambda_1, \dots, \lambda_d)$.
The observed corrupted vector is $y \triangleq x + \epsilon$.
The directional stability statistic $\Delta$ is defined as the cosine similarity:
$$\Delta \triangleq \cos(x, y) = \frac{\langle x, x + \epsilon \rangle}{\|x\|_2 \|x + \epsilon\|_2} = \frac{r^2 + \langle x, \epsilon \rangle}{r \|x + \epsilon\|_2}.$$

Decompose $\epsilon$ into parallel and orthogonal components relative to $\hat{x}$:
$$\epsilon = a \hat{x} + b, \quad a \triangleq \langle \hat{x}, \epsilon \rangle \in \mathbb{R}, \quad b \triangleq (I_d - \hat{x}\hat{x}^\top)\epsilon \in \mathbb{R}^d.$$
Under $\epsilon \sim \mathcal{N}(0, \sigma^2 I_d)$:
$$a \sim \mathcal{N}(0, \sigma^2), \quad b \sim \mathcal{N}(0, \sigma^2(I_d - \hat{x}\hat{x}^\top)), \quad a \perp b.$$
Therefore, the squared orthogonal norm satisfies:
$$\frac{\|b\|_2^2}{\sigma^2} \sim \chi_{\nu}^2, \quad \nu \triangleq d - 1.$$

---

## 2. Theorem A: Exact Distribution of Cosine Stability

### Theorem Statement
Let $x \in \mathbb{R}^d$ ($d \ge 2$) with $\|x\|_2 = r > 0$, and let $\epsilon \sim \mathcal{N}(0, \sigma^2 I_d)$. Then the directional stability statistic $\Delta = \cos(x, x+\epsilon)$ satisfies:
$$\Delta = \frac{T}{\sqrt{T^2 + \nu}},$$
where $\nu = d - 1$ and $T$ is distributed as a noncentral Student's $t$-variable:
$$T \sim t'\left(\nu = d-1, \; \lambda = \frac{r}{\sigma}\right).$$

### Proof
1. Express $\Delta$ in terms of $a$ and $b$:
   $$\Delta = \frac{r(r + a)}{r \sqrt{(r + a)^2 + \|b\|_2^2}} = \frac{r + a}{\sqrt{(r + a)^2 + \|b\|_2^2}}.$$
2. Normalize numerator and denominator by $\sigma$:
   $$\Delta = \frac{(r + a)/\sigma}{\sqrt{\frac{(r + a)^2}{\sigma^2} + \frac{\|b\|_2^2}{\sigma^2}}}.$$
3. Define the standardized variables:
   $$Z \triangleq \frac{a}{\sigma} \sim \mathcal{N}(0, 1), \quad U \triangleq \frac{\|b\|_2^2}{\sigma^2} \sim \chi_{\nu}^2, \quad \nu = d - 1.$$
   Since $a \perp b$, $Z$ and $U$ are statistically independent.
4. The numerator is $Z + \lambda$, where $\lambda \triangleq r / \sigma$.
5. Divide numerator and denominator by $\sqrt{U / \nu}$:
   $$T \triangleq \frac{Z + \lambda}{\sqrt{U / \nu}} \sim t'(\nu, \lambda).$$
6. Algebraic substitution gives:
   $$\Delta = \frac{T \sqrt{U/\nu}}{\sqrt{T^2(U/\nu) + U}} = \frac{T}{\sqrt{T^2 + \nu}}.$$
   The mapping $g(t) = \frac{t}{\sqrt{t^2 + \nu}}$ is strictly monotonically increasing from $\mathbb{R} \to (-1, 1)$. Hence, for any $\theta \in (-1, 1)$:
   $$\mathbb{P}(\Delta \le \theta) = \mathbb{P}\left(T \le \frac{\theta \sqrt{\nu}}{\sqrt{1 - \theta^2}}\right) = F_{t'(\nu, \lambda)}\left(\frac{\theta \sqrt{\nu}}{\sqrt{1 - \theta^2}}\right). \quad \blacksquare$$

### Empirical Verification
We evaluated this theorem on a $5 \times 5$ grid:
- $d \in \{16, 64, 384, 768, 1024\}$
- $r/\sigma \in \{2, 5, 10, 25, 50\}$
- Sample size: $N = 200{,}000$ per cell ($5 \times 10^6$ total samples).
- Two-sided Kolmogorov-Smirnov test against `scipy.stats.nct.cdf`:
  - **Result:** 25/25 cells passed with $p > 0.01$ (minimum $p = 0.024$, maximum $p = 0.985$).
  - Full data: `milestones/M01_theory/results/verify_thmA.json`.

---

## 3. Corollaries and Breakdown of the Manuscript Formula

### Corollary 1: Small-Noise Moment Expansion
When the noise-to-signal parameter $\kappa \triangleq \frac{(d-1)\sigma^2}{r^2} \ll 1$:
$$\mathbb{E}[\Delta] = 1 - \frac{(d-1)\sigma^2}{2r^2} + \mathcal{O}\left(\frac{\sigma^4}{r^4}\right).$$

### Corollary 2: High-Dimensional Concentration
When $d \to \infty$ with $\kappa_d \triangleq \frac{(d-1)\sigma^2}{r^2} \to \kappa \in (0, \infty)$:
$$\Delta \xrightarrow{p} \frac{1}{\sqrt{1 + \kappa}}.$$

### Corollary 3: Anisotropic Perturbation
When $\epsilon \sim \mathcal{N}(0, \Sigma)$, the energy orthogonal to $x$ is:
$$\mathbb{E}[\|b\|_2^2] = \operatorname{Tr}(\Sigma(I_d - \hat{x}\hat{x}^\top)) = \operatorname{Tr}(\Sigma) - \hat{x}^\top \Sigma \hat{x}.$$
Therefore:
$$\mathbb{E}[\Delta] = 1 - \frac{\operatorname{Tr}(\Sigma) - \hat{x}^\top \Sigma \hat{x}}{2r^2} + \mathcal{O}\left(\frac{\|\Sigma\|_F^2}{r^4}\right).$$

### Refutation of Manuscript Formula
The manuscript claimed (Eq. XVII-E):
$$\mathbb{E}[\Delta]_{\text{manuscript}} \approx 1 - \frac{\operatorname{Tr}(\Sigma)}{2r^2} = 1 - \frac{d\sigma^2}{2r^2}.$$

| Metric / Scenario | Manuscript Formula | Corrected Formula | True Monte Carlo Mean |
| :--- | :--- | :--- | :--- |
| **Dimension factor** | $d$ | $d - 1$ | $d - 1$ |
| **Parallel projection** | Ignored | Subtracted ($\hat{x}^\top\Sigma\hat{x}$) | Physically verified |
| $d=384, r/\sigma=10$ | $-0.9200$ (Negative!) | $+0.4545$ (Concentration) | $+0.4545$ |
| $d=1024, r/\sigma=2$ | $-127.0$ (Absurd!) | $+0.0624$ (Concentration) | $+0.0622$ |
| $d=384, \gamma=1.5, r/\sigma=25$ (Dominant) | $0.6928$ (Err: $0.086$) | $0.7422$ (Err: $0.036$) | $0.7787$ |

---

## 4. Theorem B: Two-Sided Mutation Detection Guarantee

### Theorem Statement
Let $H_0: \epsilon \sim \mathcal{N}(0, \sigma_0^2 I_d)$ (benign paraphrase) versus $H_1: \epsilon \sim \mathcal{N}(0, \sigma_1^2 I_d)$ with $\sigma_1 > \sigma_0$ (semantic corruption).
1. **Critical Threshold:** The decision rule $\delta_\alpha(y) = \mathbb{I}\{\Delta(x, y) < \theta_\alpha\}$ with:
   $$\theta_\alpha = \frac{t_{\nu, \lambda_0}^*}{\sqrt{(t_{\nu, \lambda_0}^*)^2 + \nu}}, \quad t_{\nu, \lambda_0}^* \triangleq F_{t'(\nu, r/\sigma_0)}^{-1}(\alpha)$$
   achieves exact type-I error rate $\alpha$.
2. **UMP Optimality:** By the Karlin-Rubin theorem, because the family of noncentral-$t$ distributions possesses a strictly Monotone Likelihood Ratio (MLR) in $T$, this test is Uniformly Most Powerful (UMP) among all invariant tests based on cosine similarity.

### Empirical Validation ($d=384, \sigma_0=0.04, \sigma_1=0.08, N=100{,}000$)
- Nominal $\alpha = 0.01$: Empirical $\mathrm{FPR} = 0.0100$, 95% CI $[0.0094, 0.0106]$ $\to$ **PASS**
- Nominal $\alpha = 0.05$: Empirical $\mathrm{FPR} = 0.0501$, 95% CI $[0.0488, 0.0514]$ $\to$ **PASS**
- Nominal $\alpha = 0.10$: Empirical $\mathrm{FPR} = 0.1010$, 95% CI $[0.0991, 0.1029]$ $\to$ **PASS**

![Theorem B ROC Curve](file:///d:/ares/milestones/M01_theory/figures/thmB_roc_theoretical.png)
![Theorem B Power Comparison](file:///d:/ares/milestones/M01_theory/figures/thmB_power_cos_vs_chi2.png)

---

## 5. Theorem C: Decision-Theoretic Verifier and Chow Abstention

### Theorem Statement
Given actions $\mathcal{A} = \{a_{\text{T}}, a_{\text{F}}, a_{\text{abstain}}\}$ and cost matrix $C(a_{\text{T}}, 0) = c_{\text{FP}}$, $C(a_{\text{F}}, 1) = c_{\text{FN}}$, $C(a_{\text{abstain}}, \cdot) = c_a$:
1. The Bayes-optimal decision rule minimizes conditional expected loss:
   $$\tau_1 \triangleq \ln\left(\frac{c_{\text{FP}} - c_a}{c_a}\right), \quad \tau_0 \triangleq \ln\left(\frac{c_a}{c_{\text{FN}} - c_a}\right).$$
2. The abstention region is non-empty if and only if:
   $$c_a < \frac{c_{\text{FP}} c_{\text{FN}}}{c_{\text{FP}} + c_{\text{FN}}}.$$
3. Under correlated evidence sources, calibrated logistic regression weights strictly outperform unweighted summation:
   $$\ell(c) = \beta_0 + \sum_{k=1}^K w_k \operatorname{LLR}_k(e_k).$$

### Empirical Validation
- Setup: $c_{\text{FP}} = 10, c_{\text{FN}} = 6, c_a = 2 \implies \tau_0 = -0.6931, \tau_1 = 1.3863$.
- Theoretical risk: $1.2180$; 2D Grid minimum: $1.2022$ (discrepancy $< 0.016$).
- Correlated sources ($\rho = 0.65, K = 4$):
  - Naive unweighted sum test log-loss: $0.4508$.
  - Learned logistic weights test log-loss: $0.3902$ ($13.44\%$ improvement).

![Theorem C Bayes Risk Surface](file:///d:/ares/milestones/M01_theory/figures/thmC_bayes_risk_surface.png)
![Theorem C Multi-Source Fusion](file:///d:/ares/milestones/M01_theory/figures/thmC_correlated_sources.png)

---

## 6. Transformer FLOP Complexity Crossover

### Derivation
For a standard transformer layer with hidden dimension $d$, sequence length $n$, and $4d$ FFN:
- Linear projections: $Q, K, V, O$ projections $= 4 \times (2nd^2) = 8nd^2$ FLOPs.
- Attention matrix: $QK^\top$ and $A V = 2 \times (2n^2d) = 4n^2d$ FLOPs.
- Feed-forward network (two matrices $d \times 4d$): $2 \times (2n \times d \times 4d) = 16nd^2$ FLOPs.
- Total forward FLOPs per layer:
  $$\text{FLOPs}(n, d) = 24nd^2 + 4n^2d.$$

### Crossover Analysis
Equating the linear projection term to the quadratic attention term:
$$24nd^2 = 4n^2d \implies n^* = 6d.$$
- For DeBERTa/RoBERTa base ($d = 768$): $n^* = 6 \times 768 = 4{,}608$ tokens.
- For typical sequence lengths $n = 512$:
  - Linear/FFN FLOPs: $24 \times 512 \times 768^2 \approx 7.25 \times 10^9$.
  - Quadratic attention FLOPs: $4 \times 512^2 \times 768 \approx 0.81 \times 10^9$ ($10.0\%$ of layer compute).
- **Refutation:** The manuscript claimed that for $n < d$, the quadratic term dominates. The opposite is true: linear/FFN projections account for $> 90\%$ of compute for all standard NLP inputs $n \le 512$.

---

## 7. Distribution Shift and Generalization Rates

### Ben-David Domain Discrepancy
Let $\mathcal{D}_S$ and $\mathcal{D}_T$ be source and target domains. By Ben-David et al. (2010):
$$\epsilon_T(h) \le \epsilon_S(h) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*,$$
where $d_{\mathcal{H}\Delta\mathcal{H}}$ is estimated via the proxy $\mathcal{A}$-distance:
$$\hat{d}_{\mathcal{A}} = 2(1 - 2 \cdot \operatorname{err}_{\text{domain}}).$$
The manuscript's assertion that ARES "explicitly minimizes discrepancy" is corrected: ARES uses representation calibration and distribution bounds to control transfer risk; it does not solve an adversarial minimax domain alignment optimization.

### Generalization Error Rate
By Rademacher complexity for bounded loss classes, generalization error scales as:
$$R(f) \le \hat{R}(f) + 2 \mathcal{R}_n(\mathcal{F}) + \sqrt{\frac{\ln(2/\delta)}{2n}} = \hat{R}(f) + \mathcal{O}\left(\frac{1}{\sqrt{|D_r|}}\right).$$
The manuscript claim of $1/|D_r|$ is corrected to the standard minimax statistical rate $1/\sqrt{|D_r|}$.

---

## 8. Provenance and Artifacts Summary

| Artifact | Type | Description |
| :--- | :--- | :--- |
| `ares_v2/theory/ares_theory.tex` | LaTeX Monograph | Consolidated formal monograph with all proofs |
| `ares_v2/theory/verify/verify_thmA.py` | Python Script | Theorem A verification ($5 \times 10^6$ samples, KS tests) |
| `ares_v2/theory/verify/verify_thmB.py` | Python Script | Theorem B verification (exact FPR, UMP power curves) |
| `ares_v2/theory/verify/verify_thmC.py` | Python Script | Theorem C verification (Chow risk surface, logistic fusion) |
| `ares_v2/theory/verify/verify_anisotropic.py` | Python Script | Corollary 3 verification (anisotropic spectral decay) |
| `milestones/M01_theory/results/verify_thmA.json` | Results JSON | Exact KS p-values across 25 grid cells |
| `milestones/M01_theory/results/verify_thmB.json` | Results JSON | Operating characteristic and power results |
| `milestones/M01_theory/results/verify_thmC.json` | Results JSON | Bayes risk and log-loss results |
| `milestones/M01_theory/results/verify_anisotropic.json` | Results JSON | Anisotropic error metrics and error reduction ratio |
| `milestones/M01_theory/tables/expansion_error.csv` | CSV Table | Small-noise vs concentration error comparison |
| `milestones/M01_theory/tables/anisotropic_verification.csv` | CSV Table | Spectral decay comparison across orientations |
| `milestones/M01_theory/figures/*.png` | Visualizations | Publication figures for ROC, risk, power, and spectra |
