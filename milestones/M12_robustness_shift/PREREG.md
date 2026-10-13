# Pre-Registration: Milestone M12 — Robustness, Distribution Shift, Stress Scenarios & Temporal External Validation

**Milestone:** M12 — Robustness, Distribution Shift, Stress Scenarios & Temporal External Validation  
**Repository:** `gurarpitzz/Ares-Cognitive-Computation`  
**Date:** October 13, 2026  
**Status:** PRE-REGISTERED  
**Commit Baseline:** `5652ae8` (`m11-complete`, `ares-v2-freeze-1`)  
**Hardware Target:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Intel Core i7-13620H, 16 GB RAM  
**Execution Environment:** Windows 11 24H2, Python 3.11.9, PyTorch 2.6.0+cu124, transformers 4.49.0, scikit-learn 1.8.0, scipy 1.17.1  
**Target Ledger Claims:** `C003`, `C019`, `C020`, `C021`, `C022`, `C023`, `C037`, `C044`, `C045`, `C056`, `C059`, `C060`, `C066`, `C073`, `C091`  

---

## 1. Scientific Objective and Context

Sections XII-D, XV-D, XXII-E/F, XXVI, and XXVII-D of the ARES manuscript make sweeping claims regarding system resilience under adversarial perturbations, cross-domain distribution shifts, crisis stress scenarios, and live social media deployment:
1. **Adversarial Resilience Claims (`C003`, `C045`, `C066`):** The manuscript claims "high resilience under stress scenarios" with an adversarial degradation $\Delta_{\text{adv}} < 0.05$ across all perturbation regimes (Fig. 42). However, no reproducible attack script, perturbation budget, or query constraint was provided in the legacy codebase (`ARES_POC`).
2. **Grover Baseline Category Error (Audit Issue, Claim `C037`):** The manuscript claimed ARES experienced only 6.3% degradation compared to Grover's 22.4%. As established in `01_MANUSCRIPT_AUDIT.md` §5, Grover's discriminator was trained for *machine-generated text detection*, not factual veracity. Comparing Grover on veracity is a category error that must be documented and revised.
3. **Discrepancy Minimization Theory Contradiction (Audit Issue M-9, Claim `C073`):** The manuscript claimed in Section XXII-E that ARES "explicitly minimizes the discrepancy term $\text{disc}_{\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T)$" from the Ben-David et al. (2010) domain adaptation bound. In reality, the ARES training objective contains no domain-adversarial loss or MMD penalty. In M12, this claim is empirically tested by measuring the proxy $\mathcal{A}$-distance ($d_{\mathcal{A}}$) between domain pairs and correlating it with transfer degradation.
4. **Ungrounded Scenario Numbers (`C019`, `C020`, `C021`, `C022`, `C023`, `C059`):** The manuscript reported exact performance metrics for real-world geopolitical conflicts and natural disasters (e.g., India–China border recall 91.1%, Assam 2022 flood accuracy 94.8%, COVID-19 F1 0.893, US/India election F1 0.875, LOSO degradation $< 4.8\%$). However, no public labelled datasets exist for the border conflict or the Assam floods.
5. **Live OSINT Deployment Claim (`C060`):** The manuscript claimed "> 88% agreement with IFCN fact-checks on live Telegram and regional X feeds". This deployment claim cannot be reproduced from static code and is replaced by a post-freeze temporal external validation protocol on ClaimReview fact-checks.

Milestone M12 replaces all legacy ungrounded numbers with pre-registered, reproducible experiments across real adversarial attacks, cross-domain transfer matrices, real scenario LOSO evaluations, controlled grounded-synthetic benchmarks (ARES-Synth v1, labelled SIMULATED), and temporal external validation.

---

## 2. Adversarial Text Attack Protocol (M12.1, Claims `C045`, `C066`)

### 2.1 Attack Recipes
To evaluate robustness against linguistic evasion, three distinct attack mechanisms are implemented:
1. **TextFooler (Synonym Substitution):** Word-level attack identifying salient words via gradient/importance ranking and replacing them with semantic neighbors from counter-fitted embeddings subject to cosine similarity threshold $\ge 0.80$ and USE semantic threshold $\ge 0.84$.
2. **DeepWordBug (Character-Level Perturbations):** Typoglycemia-based attack applying character swaps, deletions, insertions, and substitutions on top-scoring scoring tokens.
3. **Homoglyph Substitution (Visual Confusables):** Orthographic evasion replacing ASCII characters with visually indistinguishable Unicode homoglyphs (e.g., Latin `o` $\to$ Cyrillic `о`, Latin `a` $\to$ Cyrillic `а`, `e` $\to$ `е`).

### 2.2 Attack Constraints & Sampling (Audit Red Flag RF1 & RF9)
- **Target Models:**
  1. Claim-only TF-IDF + Logistic Regression baseline.
  2. DeBERTa-v3-base text veracity classifier (Track T1).
  3. Full ARES late fusion pipeline (attacks applied to input claim text; downstream evidence retrieval and module outputs recomputed on the perturbed claim).
- **Sampling:** Exactly $n = 1000$ correctly classified test instances per target drawn randomly with fixed seed (`42`). The exact instance IDs are verified to be identical across all target models (Audit Red Flag RF9).
- **Perturbation Budgets:**
  - TextFooler: Max word modification budget $\epsilon_{\text{word}} \in \{0.05, 0.10, 0.15, 0.20, 0.25, 0.30\}$; maximum query budget $Q_{\max} = 100$.
  - DeepWordBug: Max character modification budget $\epsilon_{\text{char}} \in \{0.05, 0.10, 0.15, 0.20, 0.25, 0.30\}$.
  - Homoglyphs: Substitution rate $r_{\text{homo}} \in \{0.05, 0.10, 0.15, 0.20, 0.25, 0.30\}$.
- **Metrics:**
  - Attack Success Rate: $\text{ASR} = \frac{\sum_{i=1}^n \mathbb{I}(f(x_{\text{adv}}^{(i)}) \ne y_i)}{n}$
  - Accuracy Under Attack: $\text{Acc}_{\text{adv}} = 1 - \text{ASR}$
  - Macro-F1 Under Attack ($F1_{\text{adv}}$)
  - Adversarial Degradation: $\Delta_{\text{adv}} = F1_{\text{clean}} - F1_{\text{adv}}$
  - Perturbation-Strength Curve: $F1_{\text{adv}}$ plotted against perturbation budget $\epsilon \in [0.05, 0.30]$, generating `figures/fig42_adv_strength_curve.png` (replacing manuscript Fig. 42).
  - All metrics reported with 95% percentile bootstrap confidence intervals ($B = 1000$).

---

## 3. Adversarial Training Ablation Protocol (M12.2, Claim `C056`)

### 3.1 Objective Formulation
In accordance with Equations 16–17 of the manuscript and Appendix A:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{cls}}(\theta) + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}(\theta)$$
where $\mathcal{L}_{\text{adv}}$ is Virtual Adversarial Training (VAT) in the normalized token embedding space:
$$r_{\text{adv}} = \epsilon \cdot \frac{g}{\|g\|_2}, \quad g = \nabla_{r} \text{KL}(p_\theta(y \mid x) \parallel p_\theta(y \mid x + r))$$
with perturbation radius $\epsilon = 1.0 \times 10^{-5}$.

### 3.2 Hypothesis & Experimental Design
- **Comparison:** Models trained with declared parameter $\lambda_{\text{adv}} = 0.30$ versus baseline empirical risk minimization ($\lambda_{\text{adv}} = 0.00$).
- **Replication:** $K = 10$ independent random seeds ($s \in \{0, \dots, 9\}$) on Track T1 text benchmark.
- **Evaluation:** Clean Macro-F1, Under-Attack Macro-F1 (under TextFooler budget $\epsilon = 0.15$), and Robustness Gain:
  $$\Delta_{\text{robust}} = F1_{\text{adv}}(\lambda_{\text{adv}}=0.3) - F1_{\text{adv}}(\lambda_{\text{adv}}=0.0)$$
- **Ledger Reconciliation (`C056`):** Manuscript claimed $\Delta F1 = 0.036 \pm 0.004$ and $+0.061$ under GPT-generated text. The measured empirical gain and standard error will directly update `plan/claims_ledger.csv`.

---

## 4. LLM-Generated Misinformation & Synthetic False-Claim Injection (M12.3, Claims `C037`, `C091`)

### 4.1 Evaluation on Machine-Generated Text (`C037`)
- **Evaluation Sets:** Pairwise evaluation of human-written claims versus machine-generated synthetic claims from public LLM misinformation benchmarks (LLMFake / GPT-generated paraphrases and hallucinated claims).
- **Grover Baseline Category Clarification:**
  - Grover's generator/discriminator architecture was specifically designed for *neural text provenance detection* (identifying whether text was written by a human or generated by Grover/GPT), not factual truth-checking.
  - We evaluate Grover discriminator strictly on its native task (machine text detection) and clarify that its reported 22.4% drop under adversarial veracity tasks was an artifact of testing a provenance detector out-of-domain.
  - The manuscript claim `C037` is marked **REVISED** with explicit provenance vs veracity task demarcation.

### 4.2 Synthetic False-Claim Injection Experiment (`C091`)
- **Injection Formulation:** Let $T$ be the genuine training set and $F_s$ be synthetic false claims generated via LLM prompt inversion. The injection rate is defined as:
  $$r_{\text{inj}} = \frac{|F_s|}{|T| + |F_s|} \in \{0.00, 0.05, 0.10, 0.20\}$$
- **Training & Testing:** Veracity models are trained with augmented datasets at each injection rate and evaluated on held-out **real-world test sets** ($K = 5$ seeds).
- **Metric:** Macro-F1 on real test data, False Positive Rate (FPR), and calibration error (ECE). We test whether synthetic claim injection improves out-of-domain decision boundaries or induces harmful label distortion.

---

## 5. Cross-Domain Generalization & Ben-David Discrepancy (M12.4, Claims `C044`, `C073`)

### 5.1 Domain Matrix
We pre-register an exhaustive cross-domain evaluation across 4 primary domains and 3 Tier-2 generalization domains:
1. **Politics:** LIAR benchmark (US political discourse)
2. **Health:** Constraint-COVID + CoAID (epidemiological and health misinformation)
3. **Rumours / Social Crisis:** PHEME (rumour threads across breaking news events)
4. **Multilingual:** X-FACT (multilingual claims across 25 languages)
5. **Tier-2 Domains:** SciFact / CLIMATE-FEVER (scientific/climate claims), PUBHEALTH (biomedical claims), IFND / FactDrill (regional Indian news).

### 5.2 Deduplication Pre-processing (Audit Red Flag RF4)
Before cross-domain evaluation, all datasets containing PolitiFact claims (LIAR, UPFD, X-FACT) undergo exact and near-duplicate deduplication (MinHash Jaccard threshold $\ge 0.85$) to guarantee that no test claim in domain $j$ appeared in the training set of domain $i$.

### 5.3 Proxy $\mathcal{A}$-Distance Computation (Audit Issue M-9, Claim `C073`)
For each ordered pair of domains $(D_S, D_T)$:
1. Extract frozen representations $\mathbf{h} \in \mathbb{R}^{768}$ from the frozen DeBERTa-v3-base encoder.
2. Form a binary classification task to distinguish source instances ($y=0$) from target instances ($y=1$).
3. Train a linear SVM domain discriminator $h \in \mathcal{H}$ on a balanced held-out split (5-fold cross-validation; Audit Red Flag RF5).
4. Compute the cross-validation error rate $\epsilon(h)$ and proxy $\mathcal{A}$-distance:
   $$\hat{d}_{\mathcal{A}}(D_S, D_T) = 2(1 - 2 \cdot \epsilon(h))$$
5. Measure the empirical transfer degradation:
   $$\Delta F1(D_S \to D_T) = F1(D_T \mid \text{train on } D_T) - F1(D_T \mid \text{train on } D_S)$$
6. Compute the Spearman rank correlation $r_s$ between $\hat{d}_{\mathcal{A}}$ and $\Delta F1$ with 95% bootstrap confidence intervals.
7. **Theory Correction (`C073`):** Confirm that while ARES does *not* explicitly minimize discrepancy in its loss, the Ben-David bound $\epsilon_T(h) \le \epsilon_S(h) + \frac{1}{2} d_{\mathcal{A}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*$ accurately bounds empirical cross-domain degradation.

---

## 6. Leave-One-Scenario-Out (LOSO) Stress Scenarios (M12.5, Claims `C019`, `C020`, `C023`, `C059`)

### 6.1 Real Scenario Definitions (Audit Red Flag RF3)
To eliminate post-hoc scenario selection, the evaluated real-world scenarios are pre-registered unconditionally:
- **Scenario S_COVID (COVID-19 Infodemic):** Health claims from Constraint-COVID and CoAID.
- **Scenario S_ELECTION_US (US Elections):** Pre-registered keyword filter on LIAR (e.g., `election`, `ballot`, `presidential`, `campaign`, `voting`, `trump`, `biden`, `obama`, `democrat`, `republican`).
- **Scenario S_CRISIS_PHEME (Breaking-News Crisis Events):** PHEME rumour clusters across 5 canonical breaking news crises:
  1. Charlie Hebdo shooting (Paris, 2015)
  2. Sydney siege (Lindt Cafe, 2014)
  3. Ferguson unrest (2014)
  4. Ottawa parliament shooting (2014)
  5. Germanwings flight crash (2015)
- **Scenario S_INDIA (Indian Fact-Check Corpus):** Regional Indian election and public interest claims from IFND and FactDrill.

### 6.2 LOSO Evaluation Protocol
For each scenario $s \in \mathcal{S}$:
1. Train ARES on all scenarios excluding $s$: $\mathcal{D}_{\text{train}} = \bigcup_{s' \ne s} \mathcal{D}_{s'}$.
2. Test on held-out scenario $\mathcal{D}_s^{\text{test}}$ across $K = 5$ random seeds.
3. Compare with in-scenario benchmark trained on $\mathcal{D}_s^{\text{train}}$.
4. Compute absolute scenario degradation:
   $$\Delta_{\text{LOSO}}(s) = F1_{\text{in-scenario}}(s) - F1_{\text{held-out}}(s)$$
5. **Ledger Reconciliation (`C059`):** Manuscript claimed mean LOSO degradation $< 4.8\%$. Report measured empirical degradation across all scenarios.

---

## 7. Real-Data Search & ARES-Synth Controlled Grounded Synthetic Stress Evaluation (M12.6, M12.6b, Claims `C020`-India, `C021`, `C022`)

### 7.1 Search for Missing Scenario Datasets (M12.6)
We conduct and document a comprehensive search across public repositories (HuggingFace, Zenodo, Kaggle, Harvard Dataverse) and Indian fact-checking APIs for labelled datasets on:
- India–China border conflict (Galwan 2020, Tawang 2022)
- Assam 2022 floods
- 2024 Indian general election

The search methodology, queries, URLs, and findings are logged in `milestones/M12_robustness_shift/results/scenario_data_search.md`. If a public labelled dataset with $n \ge 100$ items exists with verified ground-truth labels, it takes precedence. Otherwise, the claims are evaluated on ARES-Synth v1.

### 7.2 ARES-Synth v1 Construction & Rules (Audit Red Flags RF7, RF8)
ARES-Synth v1 is a controlled, grounded-synthetic stress benchmark built from frozen source documents:
- **Scenario S1 (Border Conflict):** Grounded in official MEA/MFA releases and Wikipedia snapshots (May 2020 – Jan 2023).
- **Scenario S2 (Assam Floods 2022):** Grounded in ASDMA daily flood bulletins and ReliefWeb situation reports (Apr – Oct 2022).
- **Scenario S3 (Indian Election 2024):** Grounded in ECI official notifications and press releases.
- **Deterministic Perturbation Engine:**
  All labels are generated by deterministic code perturbations (never an LLM):
  1. `entity_swap`: Subject/object swapped with an entity of the same type.
  2. `location_swap`: Swapped with a distinct geographic location $\ge 50$ km away.
  3. `number_change`: Numerical value scaled by factor $\in \{0.25, 0.5, 2.0, 3.0, 10.0\}$.
  4. `date_shift`: Shifted by $\pm \{7\text{ days}, 1\text{ month}, 1\text{ year}\}$.
  5. `negation`: Polarity inverted via curated antonym dictionary.
  6. `misattribution`: Source authority swapped.
  7. `out_of_context`: Real historical fact mapped to an unrelated event.
  8. `fabricated_event`: Event combination absent from evidence corpus ($\to$ NEI).
- **Surface Realization:** 4 registers (news headline, social media post, forwarded message, official statement) with slot preservation validation and dual NLI filtering.
- **Integrity Constraints:**
  - Zero training, tuning, or threshold selection on ARES-Synth (Audit Red Flag RF7).
  - Evaluated **test-only** on frozen models (`ares-v2-freeze-1`).
  - Every table, figure, and report carries the caption: `SIMULATED — grounded synthetic (ARES-Synth v1)` (Audit Red Flag RF8).
  - Claims `C021`, `C022`, and `C020-India` are updated in the ledger with the qualifier *"on a controlled grounded-synthetic benchmark"*.

---

## 8. Temporal External Validation Protocol (M12.7, Track T11, Claim `C060`)

### 8.1 Track T11 Specification
To replace the manuscript's ungrounded live OSINT claim (`C060`), Track T11 pre-registers a reproducible post-freeze temporal validation protocol:
1. **Freeze Milestone Reference:** Model weights and retrieval indexes are frozen at tag `ares-v2-freeze-1` (Oct 13, 2026).
2. **Collection Source:** Google Fact Check Tools API querying ClaimReview records from certified IFCN signatories (PolitiFact, FactCheck.org, BOOM, Alt News, Full Fact, AFP Fact Check, Washington Post).
3. **Temporal Constraint:** All evaluated claims must have publication date $t_{\text{claim}} > t_{\text{freeze}}$.
4. **Evidence Temporal Constraint:** In retrieval mode, evidence retrieval is restricted to documents with publication timestamps $t_{\text{evidence}} \le t_{\text{claim}}$ to eliminate future hindsight leakage.
5. **Rating Mapping:** Pre-registered deterministic mapping from publisher verdict strings to binary veracity labels (`False`, `True`, `Uncertain`). Ambiguous ratings are excluded and logged.
6. **Deduplication:** Strict deduplication against all training data across all pre-registered tracks T1–T9.

---

## 9. Media Robustness Summary & "High Resilience" Replacement Statement (M12.8, M12.9, Claim `C003`)

### 9.1 Cross-Modal Robustness Synthesis
We collate empirical degradation numbers across modalities into a unified table (`tables/table_media_robustness.csv`):
- **Vision Deepfakes (M07):** JPEG compression ($Q \in \{30, 50, 70, 90\}$), Gaussian blur ($\sigma \in \{1, 2, 3\}$), and cross-dataset evaluation (Celeb-DF $\to$ FF++).
- **Audio Spoofing (M08):** Additive Gaussian noise ($\text{SNR} \in \{10, 20, 30\}\text{ dB}$), MP3/AAC codec re-encoding, and silence trimming shortcut evaluation.
- **Text Adversarial (M12.1):** TextFooler, DeepWordBug, Homoglyphs.

### 9.2 Evidence-Based Replacement Statement (`C003`)
Claim `C003` ("High resilience under stress scenarios") is replaced in the manuscript with a quantified, scientifically defensible statement:
> *"Under adversarial word-level synonym substitution (TextFooler, 15% budget), ARES late fusion demonstrates an empirical macro-F1 degradation of $\Delta_{\text{adv}} = [X.XX]$, outperforming claim-only text encoders ($\Delta = [X.XX]$). Under leave-one-scenario-out crisis shifts, mean performance decreases by $[\Delta\%]$, with domain degradation closely bounded by proxy $\mathcal{A}$-distance ($r_s = [X.XX]$). On controlled grounded-synthetic conflict and disaster benchmarks (ARES-Synth v1, SIMULATED), ARES achieves macro-F1 of $[X.XX]$ on border conflict claims and $[X.XX]$ on flood claims."*

---

## 10. Audit Gates & Red Flags

- **RF1 (Test-Only Attacks):** Adversarial attacks are evaluated strictly on test instances, never training or validation sets.
- **RF2 (Temporal Purity):** Temporal external validation contains no claims reviewed prior to the model freeze date; evidence retrieval enforces $t_{\text{evidence}} \le t_{\text{claim}}$.
- **RF3 (Fixed Scenarios):** All LOSO scenario boundaries and keyword rules are fixed in this pre-registration; no post-hoc scenario cherry-picking.
- **RF4 (Deduplication):** Cross-domain benchmarks deduplicate cross-dataset PolitiFact overlaps prior to testing.
- **RF5 (Held-Out Proxy $\mathcal{A}$-Distance):** The domain discriminator is trained and evaluated on 5-fold cross-validation splits of held-out representations.
- **RF6 (ARES-Synth Isolation):** Zero training, fine-tuning, or threshold selection occurs on ARES-Synth. Manifest hash is locked.
- **RF7 (Strict SIMULATED Labeling):** ARES-Synth results are never pooled into monolithic real-data numbers. Every synthetic table carries the caption `SIMULATED — grounded synthetic (ARES-Synth v1)`.
- **RF8 (Identical Attack Samples):** Text attack samples are identical across all target models (same $n=1000$ item IDs, verified by SHA-256 hash).
