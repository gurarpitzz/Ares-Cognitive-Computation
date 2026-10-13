# ARES Legacy POC — Audit and Autopsy Notes

**Reference Path:** `ares_v2/legacy/ares_poc/`  
**Source Repository:** `github.com/gurarpitzz/ARES_POC` (state as of 2026-10-13)  
**Status:** ARCHIVED / READ-ONLY REFERENCE. NEVER IMPORTED BY `src/ares/`.

---

## 1. Overview of Legacy Codebase
The original proof-of-concept repository consisted of approximately 30 KB of Python implementing a minimal web-retrieval pipeline (`Truth'(C) = Σ S·N·W / Σ S`). It attempted to demonstrate evidence retrieval and natural language inference over arbitrary user claims.

---

## 2. Identified Deficiencies, Anti-Patterns, and Non-Reproducibility

### 2.1 Live Web Retrieval & Evidence Leakage
* **Flaw:** Evidence was scraped dynamically from the live open web via search engines at runtime.
* **Impact:** 
  1. Complete loss of temporal reproducibility: search engine indexes change continuously.
  2. Severe label leakage: querying fact-checked claims (e.g. from LIAR or PolitiFact) frequently returned the fact-checking article itself, leaking the ground truth label into the evidence pool.
* **Resolution in v2:** All evidence engines operate on frozen, snapshot-indexed corpora (FEVER 2017 Wikipedia dump, AVeriTeC knowledge store) with pre-indexed BM25 and dense FAISS vectors.

### 2.2 Silent Fallbacks & Toy Datasets
* **Flaw:** `evaluate.py` wrapped dataset downloads in try-except blocks that quietly fell back to a hardcoded list of 2 dummy claims upon failure.
* **Impact:** An evaluation run with broken downloads would exit with code 0 and report artificial metrics without alerting the researcher.
* **Resolution in v2:** Rule 3 of the Gemini Operating Protocol strictly forbids silent fallbacks. All data loaders fail loudly, and AST scanner `tests/integrity/test_no_silent_fallback.py` blocks silent except blocks.

### 2.3 Mathematically Unbounded "Confidence"
* **Flaw:** The legacy code calculated confidence as:
  $$\text{Conf}(C) = |\text{Truth}'(C)| \cdot \log(1 + M)$$
  where $M$ is the number of retrieved evidence sentences.
* **Impact:** For $M = 7$ and $|\text{Truth}'| \approx 1$, this yielded values exceeding $2.08$, yet was described in the manuscript as a "calibrated confidence probability".
* **Resolution in v2:** Replaced by calibrated probabilities derived from logistic likelihood ratios (LLR) and Platt / temperature scaling, verified via Expected Calibration Error (ECE) and Brier scores (M05).

### 2.4 Arbitrary Decision Thresholds
* **Flaw:** Verdicts were assigned using an arbitrary hardcoded threshold of $\text{Truth}' < -0.4$, with no theoretical derivation or empirical risk minimization.
* **Resolution in v2:** Decision theory framing using Chow's rule with explicit misclassification vs. abstention costs ($\tau_0, \tau_1$).

### 2.5 Primitive Heuristic Weighting
* **Flaw:** Credibility weighting $W(E)$ relied on a hard-coded dictionary of domain keywords.
* **Resolution in v2:** Domain credibility priors grounded in published empirical source-quality datasets (Lin et al. 2023, PNAS Nexus).

### 2.6 Missing Modules
* **Deficiency:** The legacy POC lacked training code for DeBERTa, deepfake detection, audio forensics, bot/behavior graphs, geopolitical context, track-level fusion, and statistical testing.
* **Resolution in v2:** Completely engineered from scratch across milestones M01–M16.
