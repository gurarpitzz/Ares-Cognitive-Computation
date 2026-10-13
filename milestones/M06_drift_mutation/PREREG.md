# Milestone M06 Pre-Registration — Claim Drift & Mutation

**Milestone:** M06 — Claim Drift & Mutation: Test Theory on Real Embeddings + Claim Memory  
**Author:** Antigravity Autonomous Research Agent  
**Date:** 2026-10-13  
**Status:** FROZEN BEFORE EXECUTION  
**Target Ledger IDs:** C011, C012, C070 (empirical validation), C074, C085, C086  

---

## 1. Operational Definitions

The manuscript's notion of "mutation" conflates semantic paraphrasing, stylistic drift, and adversarial truth-reversal. We establish strict operational definitions:

1. **$H_0$ Pairs (Benign Evolution / Paraphrase):**
   - Pairs preserving core factual proposition.
   - Grounded in canonical PAWS (`label = 1`) and STS-B (`score >= 4.5 / 5.0`).
   - Represents legitimate linguistic variation without veracity change.

2. **$H_1$ Pairs (Semantic Mutation / Factual Disruption):**
   - **Family A (Lexical Disruption):** PAWS pairs with high surface overlap but altered meaning (`label = 0`).
   - **Family B (Narrative Shift):** Claims from distinct fact-checking articles within the same thematic cluster (e.g. conflicting COVID or political claims).
   - **Family C (Controlled Synthetic Perturbations):** Seeded entity swaps, number alterations, negation insertions, and temporal displacements (strictly labelled `SIMULATED`).

---

## 2. Pre-Registered Hypotheses

* **H6.1 (Gaussian / Isotropic Assumptions):**
  The perturbation vector $\epsilon = x' - x$ on $H_0$ sentence embeddings will be evaluated for multivariate normality and isotropy.
  * *Anticipated Empirical Finding:* Transformer sentence embeddings are well-known to suffer from dimensional anisotropy (cone effect); the isotropic hypothesis ($\Sigma = \sigma^2 I$) is expected to be rejected ($p < 0.001$). We pre-register reporting the effective rank, condition number, and ratio $\text{Tr}(\hat{\Sigma}) / (d \cdot \lambda_{\max})$.

* **H6.2 (Theoretical vs Empirical ROC Alignment):**
  The theoretical ROC predicted by Theorem A and Theorem B (using parameters fitted on training pairs) will track the empirical ROC on held-out test pairs within $|\Delta\text{AUC}| \le 0.05$.

* **H6.3 (Detection Power: Distance vs Cosine):**
  On normalized unit-sphere embeddings, Euclidean distance squared $\|\epsilon\|^2 = 2(1 - \cos\theta)$ has equivalent rank order to cosine; on unnormalized raw embeddings, Euclidean distance incorporates magnitude variance. We evaluate whether raw distance provides higher power than cosine under heterogeneous sentence lengths.

* **H6.4 (Temporal Claim Memory Utility):**
  A chronological memory store of previously verified claims with strict non-leakage ($t_{\text{memory}} < t_{\text{query}}$) provides significant predictive signal for incoming claims (MAP@5 $> 0.70$ on near-duplicate matching).

* **H6.5 (Verification Log-Odds Drift):**
  Along chronological mutation chains, verification log-odds drift negatively across true mutation steps:
  $$\Delta L_t = L(c_{t+1}) - L(c_t) = -\kappa \cdot \mathbb{I}(\text{mutation}) + \eta_t, \quad \kappa > 0$$

---

## 3. Red Flags & Verification Gates

* **RF1 (Temporal Leakage):** The claim memory module strictly forbids accessing claims where $t_{\text{memory}} \ge t_{\text{query}}$. Unit tests must enforce this condition.
* **RF2 (Article-Level Split Disjointness):** The same fact-check cluster / article ID must never appear across train and test pair splits.
* **RF3 (Threshold Selection):** All decision thresholds ($\theta_{\text{detect}}, \theta_{\text{match}}$) are tuned strictly on validation splits.
* **RF4 (Non-Circular Definitions):** $H_1$ mutations are defined by external annotation (PAWS/STS ground truth), never by the similarity metric being tested.
* **RF5 (Simulated Provenance):** Any synthetic mutation chains are explicitly marked `SIMULATED` and reported separately from human-authored benchmark pairs.
