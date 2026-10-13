# M04 Pre-Registration — Evidence Engine R·S·N·W + Logic Layer

**Milestone:** M04 — Evidence Engine R·S·N·W + Logic Layer  
**Date:** 2026-10-13  
**Status:** PRE-REGISTERED  
**Target Ledger Claims:** C018 (Layer-wise pipeline metrics / Table XV), C040 (Explainable trace), C067 (Showcase trace reproduction), C078 (Calibrated confidence score fix)  
**Primary Metrics:** FEVER 1.0 Label Accuracy and Macro-F1 (with 95% percentile bootstrap CIs, B = 10,000)  
**Secondary Metrics:** Evidence Retrieval Recall@{1, 5, 10}, Mean Reciprocal Rank (MRR), Calibration Error (ECE)  
**Compute Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Intel Core i7-13620H (16 threads)  

---

## 1. Scope & Objective

Implement and rigorously evaluate the complete ARES Evidence Retrieval & Scoring Pipeline:
$$\text{Truth}'(C) = \frac{\sum_{i=1}^K S_i \cdot N_i \cdot W_i}{\sum_{i=1}^K S_i}$$
where:
- $R$: Candidate evidence retrieval over frozen corpora (FEVER Wikipedia knowledge source).
- $S$: Semantic relevance / passage ranking score.
- $N$: Natural Language Inference stance score ($N \in [-1, +1]$).
- $W$: Source credibility weighting ($W \in [0, 1]$).

Furthermore, build the Symbolic Logic Layer (`ares.logic`) for numerical, date, and relational consistency checking to catch reasoning errors that neural encoders miss.

---

## 2. Pre-Registered Hypotheses

### H4.1: NLI Model Comparison (Uncontaminated Setting)
- **Hypothesis:** Upgrading the NLI backbone from BART-large-MNLI (POC) to a modern de-biased encoder trained strictly without FEVER data (e.g. RoBERTa-large-MNLI or DeBERTa-v3-base-MNLI) achieves higher 3-way label accuracy on FEVER under oracle gold-evidence inputs.
- **Protocol:** Models must be audited to verify that training data did not include FEVER-NLI (Red Flag RF1 compliance).

### H4.2: Hybrid Retrieval Superiority
- **Hypothesis:** Hybrid reciprocal-rank fusion of lexical BM25 (`rank-bm25`) and dense vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`) achieves strictly higher Evidence Recall@5 than BM25 alone or dense retrieval alone on FEVER claims.

### H4.3: Soft-N vs Hard-N Functional
- **Hypothesis:** The continuous soft stance functional ($N = P(\text{SUPPORTS}) - P(\text{REFUTES})$) yields higher Macro-$F_1$ and lower Expected Calibration Error than the discrete hard-N formulation ($N \in \{-1, 0, +1\}$) across decision thresholds.

### H4.4: Credibility Weighting (W) Utility
- **Hypothesis:** Applying continuous domain credibility weights $W_i \in [0, 1]$ based on empirical journalistic standards increases precision on conflicting evidence sources compared to unweighted averaging ($W=1.0$).

---

## 3. Integrity Gates & Negative Constraints

1. **RF1 (No NLI Contamination):** NLI checkpoints evaluated for headline performance must not be trained on FEVER. Any FEVER-trained model evaluated will be explicitly marked as `CONTAMINATED_UPPER_BOUND`.
2. **RF2 (Fact-Check Exclusion):** Web retrieval must strictly filter out known fact-checking domains (Snopes, PolitiFact, FullFact, BoomLive, AltNews, FactCheck.org) to prevent target label leakage.
3. **RF3 (Temporal Safeguard):** No evidence with a timestamp after the claim timestamp may be admitted.
4. **RF4 (Strict Split Isolation):** All threshold tuning (verdict thresholds $\tau_{\text{true}}, \tau_{\text{false}}$) must be performed strictly on the validation set (`val`). The official test set (`test`) is evaluated strictly once.
