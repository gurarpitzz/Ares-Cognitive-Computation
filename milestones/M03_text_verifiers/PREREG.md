# M03 Pre-Registration — Text Baselines & Layer-1 Text Verifiers

**Milestone:** M03 — Text Baselines & Layer-1 Text Verifiers  
**Date:** 2026-10-13  
**Status:** PRE-REGISTERED  
**Target Ledger Claims:** C038 (Layer-1 text models), C039 (multilingual coverage), C050 (hyperparameter declarations), C064 (baseline scoping), C065 (empirical SOTA comparison)  
**Primary Verification Metrics:** Macro-F1 on official test splits (with 95% percentile bootstrap CIs, B = 10,000)  
**Secondary Metrics:** Accuracy, per-class Precision/Recall/F1, AUC-ROC, Expected Calibration Error (ECE)  
**Compute:** RTX 4060 Laptop (Intel Core i7-13620H 16 threads, CUDA 12.4, 8 GB VRAM)  

---

## 1. Scope & Objective
Establish empirical, reproducible benchmarks for text-only claim veracity across Tier-1 datasets (LIAR, Constraint-COVID, CoAID, X-FACT). Compare classical statistical baselines (Majority, Stratified Random, TF-IDF + Logistic Regression, TF-IDF + Linear SVM) against transformer encoders (RoBERTa-base, DeBERTa-v3-base, mDeBERTa-v3-base) under strictly matched hyperparameter protocols. Cache all instance-level test predictions for downstream multimodal fusion (M05, M11).

---

## 2. Pre-Registered Hypotheses

### H3.1: Neural Encoder Superiority over Linear Surface Features
- **Hypothesis:** Fine-tuned DeBERTa-v3-base achieves strictly higher test macro-F1 than tuned TF-IDF + Logistic Regression across all English veracity benchmarks (LIAR, Constraint-COVID, CoAID).
- **Test:** Paired bootstrap difference $\Delta \text{Macro-F1} > 0$ with one-sided $p < 0.05$ (Holm-Bonferroni family-wise error rate control).

### H3.2: Encoder Architecture Ranking
- **Hypothesis:** Exploratory comparison among {BERT-base, RoBERTa-base, DeBERTa-v3-base}. We test whether DeBERTa-v3-base's disentangled attention yields higher accuracy on complex political claims (LIAR) compared to standard BERT/RoBERTa representations.

### H3.3: Multilingual Generalization on X-FACT
- **Hypothesis:** mDeBERTa-v3-base achieves macro-F1 strictly above the majority-class baseline across at least 10 non-English languages with test support $N \ge 50$.

---

## 3. Experimental Protocols & Hyperparameters

### 3.1 Classical Baselines Protocol
1. **Majority Class:** Predicts the most frequent training label for all test instances.
2. **Stratified Random:** Predicts labels sampled according to training class prior probabilities.
3. **TF-IDF + Logistic Regression:**
   - Feature extraction: Word n-grams (1, 2) and character n-grams (3, 5), sublinear TF scaling, max features 50,000.
   - Regularization: L2 penalty, parameter $C \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$ tuned strictly on the validation set.
4. **TF-IDF + Linear SVM:**
   - LinearSVC with hinge loss, $C \in \{0.01, 0.1, 1.0, 10.0\}$ tuned strictly on the validation set.

### 3.2 Transformer Fine-Tuning Protocol (Claim C050 Grounding)
- **Model Checkpoints:**
  - `microsoft/deberta-v3-base`
  - `roberta-base`
  - `microsoft/mdeberta-v3-base` (multilingual)
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$, weight decay = $0.01$).
- **Learning Rate:** $2 \times 10^{-5}$ peak learning rate.
- **Schedule:** 10% linear warm-up followed by cosine learning rate decay.
- **Batch Size:** Effective batch size 32 (micro-batch 16 with gradient accumulation 2).
- **Epochs:** Up to 15 epochs with early stopping based on validation macro-F1 (patience = 3 epochs).
- **Precision:** `bfloat16` autocast acceleration.
- **Sequence Length:** Dynamic length bucketing, capped at 512 tokens.
- **Seeds:** Pinned seeds $S \in \{42, 43, 44, 45, 46, 47, 48, 49, 50, 51\}$ ($K=10$ runs per benchmark).

---

## 4. Integrity and Leakage Prevention
1. **Blind Input Policy:** Only the claim statement text is fed into the models. All metadata columns (speaker, party affiliation, credit history, source URL, justifications) are strictly withheld.
2. **Test Set Access:** The test split is evaluated strictly once per seed after freezing the model with the highest validation macro-F1.
3. **Instance-Level Predictions:** Every experiment writes a Parquet file containing `item_id`, `gold_label`, `predicted_label`, and class probabilities.
