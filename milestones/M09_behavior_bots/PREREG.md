# Pre-Registration: Milestone M09 — Behavioral Analysis, Bot Detection, and Social Coordination Graphs

**Date:** 2026-10-13  
**Status:** PRE-REGISTERED  
**Commit Target:** `M09: Pre-register behavioral analysis hypotheses, models, and protocols`  
**Milestone Directory:** `milestones/M09_behavior_bots/`  
**Target Ledger Claims:**  
- `C080` (Section VII-B, Fig. 11): Posting interval distribution for bot vs human accounts (currently UNVERIFIED / illustrative).  
- `C081` (Section VII-C, Fig. 12): Social graph clustering and community structure (currently UNVERIFIED / illustrative).  
- Prepares behavioral input signals for multimodal fusion in Milestone M11 (Claims `C043`, `C048`, `C053`).

---

## 1. Objectives & Scope

In the ARES architecture, Section VII specifies a 4-pillar behavioral analysis pipeline designed to detect automated accounts (social bots) and coordinated propagation campaigns:
1. **Stylometry & Authorship Anomaly (Section VII-A):**
   - Character n-gram frequencies ($n \in \{2, 3, 4\}$).
   - Punctuation variance $\sigma_p^2 = \frac{1}{N}\sum_{i=1}^N (p_i - \mu_p)^2$.
   - Yule's Characteristic K metric:
     $$K = 10^4 \cdot \frac{\sum_{i=1}^\infty i^2 f_i - N}{N^2}$$
     where $f_i$ is the number of distinct word types appearing $i$ times, and $N$ is total tokens.
   - One-Class Support Vector Machine (OCSVM) trained strictly on verified human accounts ($y=0$) to flag anomalous, non-human posting styles.
2. **Temporal Posting Patterns & HMM Log-Likelihood Ratio (Section VII-B):**
   - Inter-post intervals $\Delta t_i = t_{i+1} - t_i$.
   - Shannon entropy of binned posting intervals: $H(\Delta t) = -\sum_b p_b \log_2 p_b$.
   - Spectral periodicity and autocorrelation peaks.
   - Hidden Markov Models (HMM with Gaussian emissions on log-intervals) fitted separately on human accounts ($\lambda_{\text{human}}$) and bot accounts ($\lambda_{\text{bot}}$).
   - Log-Likelihood Ratio (LLR) per test account:
     $$\Lambda = \log P(O \mid \lambda_{\text{bot}}) - \log P(O \mid \lambda_{\text{human}})$$
3. **Profile Metadata Features (Section VII-B/C):**
   - Followers-to-friends ratio: $\rho = \frac{\text{followers}}{\text{friends} + 1}$.
   - Account age in days, tweet generation rate ($\text{statuses}/\text{age}$), verified status, default profile flags, favourites count.
4. **Social Graph Topology & Coordination Features (Section VII-C):**
   - In-degree, out-degree, total degree centrality.
   - PageRank centrality: $PR(u) = \frac{1-d}{N} + d \sum_{v \in \mathcal{N}_{in}(u)} \frac{PR(v)}{out(v)}$.
   - Eigenvector centrality: $x_u = \frac{1}{\lambda} \sum_{v \in \mathcal{N}(u)} x_v$.
   - Local clustering coefficients: $C(u) = \frac{2 e_u}{k_u (k_u - 1)}$.
   - Louvain community detection modularity features and community membership bot proportions.

---

## 2. Benchmark Datasets

| Dataset | Modality | Samples | Role / Split | Protocol Specification |
| :--- | :--- | :--- | :--- | :--- |
| **TwiBot-20** | Social Graph + Profile Metadata | 11,826 labelled accounts, 227,979 edges | Train (8,278), Dev (2,365), Test (1,183) | Official benchmark split from Sun et al. (CIKM 2020). Graph topology evaluated transductively and inductively on held-out test accounts. |
| **Bot Detection Text Benchmark** | Tweet Stylometry & Text | 50,000 tweets (25,000 bot, 25,000 human) | Train (25,000), Val (10,000), Test (15,000) | Account-level stratified split for stylometry, n-grams, and OCSVM evaluation. |
| **Twitter-Human-Bots** | Tabular Profile Metadata | 37,438 accounts (25,013 human, 12,425 bot) | Cross-dataset Test (Held-out) | Botometer-standardized profile benchmark for cross-dataset generalization. |
| **Temporal Posting Timelines** | Timestamps & Posting Logs | Real bot vs human post timelines | Timeline analysis & HMM LLR | Inter-post interval modeling and Figure 11 replacement. |

---

## 3. Formal Hypotheses

- **H9.1 (Feature Family Superiority):**  
  Each behavioral family (Stylometry, Temporal-HMM, Profile Metadata, and Graph Centrality) evaluated independently on held-out test accounts significantly outperforms the majority baseline in Macro-$F_1$ and ROC-AUC ($p < 0.05$, paired bootstrap test with Holm-Bonferroni correction across families).
- **H9.2 (Authorship Anomaly OCSVM):**  
  The One-Class SVM anomaly detector trained strictly on human accounts ($y=0$) achieves $\text{ROC-AUC} > 0.65$ in discriminating automated bot accounts from humans without seeing any bot examples during training.
- **H9.3 (Multimodal Synergy):**  
  A fused classifier combining all four behavioral families (Stylometry + Temporal + Profile + Graph) achieves higher Macro-$F_1$ and ROC-AUC than any single behavioral family in isolation ($\Delta F_1 > 0.02$, $p < 0.05$).
- **H9.4 (Cross-Dataset Generalization):**  
  Classifiers trained on TwiBot-20 profile features retain $\text{ROC-AUC} > 0.80$ when evaluated zero-shot on the held-out Twitter-Human-Bots benchmark.

---

## 4. Models to Evaluate

1. **Majority / Stratified Random Baselines** (Theoretical zero-signal anchors).
2. **Authorship Anomaly One-Class SVM (OCSVM):** RBF kernel, $\nu \in [0.01, 0.2]$, $\gamma \in [\text{'scale'}, 0.01, 0.1]$ tuned on validation split.
3. **Inter-Post Interval Hidden Markov Model (HMM):** 2 to 4 hidden states with Gaussian emissions on log-intervals; parameters $\lambda_{\text{bot}}$ and $\lambda_{\text{human}}$ fitted exclusively on training accounts.
4. **Graph Centrality Baseline:** Logistic Regression and Gradient Boosting on degree, PageRank, eigenvector centrality, clustering coefficients, and Louvain community features.
5. **Profile Metadata Baseline:** HistGradientBoosting and Random Forest on numerical/categorical account metadata.
6. **ARES Multimodal Behavior Classifier:** Ensemble of stylometric anomaly scores, HMM log-likelihood ratio $\Lambda$, graph centrality features, and profile embeddings.

---

## 5. Metrics & Statistical Testing

- **Primary Metrics:** ROC-AUC, Equal Error Rate (EER), Macro-$F_1$, Bot-$F_1$, Accuracy.
- **Confidence Intervals:** 2000-sample percentile bootstrap 95% confidence intervals for all primary metrics.
- **Significance Tests:** McNemar's exact test for classification accuracy differences; paired bootstrap test for ROC-AUC and $F_1$ differences with Holm-Bonferroni correction for multiple comparisons.

---

## 6. Milestone-Specific Red-Flag Gates (RF1–RF6)

- **RF1 (Account-Level Split Integrity):** No account or user ID may appear in multiple splits. N-gram vocabularies, scalers, and OCSVM parameters must be fitted strictly on the training split.
- **RF2 (Temporal Horizon Non-Leakage):** When extracting temporal and interval features, no post occurring after the designated evaluation cutoff may be used.
- **RF3 (Transductive vs. Inductive Graph Clarity):** Any graph feature derived from global topology (PageRank, Louvain communities) must clearly document whether the graph contains test nodes (transductive) or only train nodes (inductive). Both settings must be evaluated.
- **RF4 (Known TwiBot-20 Benchmark Integrity):** Account labels and splits must strictly adhere to the official TwiBot-20 benchmark partition; results must be benchmarked against published literature.
- **RF5 (Domain Transfer Transparency):** Cross-dataset transfer across benchmarks must be explicitly labeled as zero-shot domain transfer with documented domain shift.
- **RF6 (Privacy & Anonymization in Figures):** All usernames, handles, and screen names displayed in publication figures (Fig. 11, Fig. 12) must be fully anonymized or masked (`User_XXXX`).

---

## 7. Expected Output Artifacts

- **Code:**
  - `ares_v2/src/ares/behavior/stylometry.py`
  - `ares_v2/src/ares/behavior/temporal.py`
  - `ares_v2/src/ares/behavior/graph.py`
  - `ares_v2/src/ares/behavior/profile.py`
  - `ares_v2/src/ares/behavior/classifier.py`
  - `ares_v2/tests/test_behavior.py`
  - `ares_v2/scripts/run_m09_evaluation.py`
- **Tables:**
  - `milestones/M09_behavior_bots/tables/bot_detection_main.csv` (.tex)
  - `milestones/M09_behavior_bots/tables/feature_family_ablation.csv` (.tex)
  - `milestones/M09_behavior_bots/tables/graph_centrality_metrics.csv` (.tex)
  - `milestones/M09_behavior_bots/tables/cross_dataset_generalization.csv` (.tex)
- **Figures:**
  - `milestones/M09_behavior_bots/figures/fig10_behavior_pipeline.png` (Architecture)
  - `milestones/M09_behavior_bots/figures/fig11_posting_intervals.png` (Claim C080)
  - `milestones/M09_behavior_bots/figures/fig12_social_graph_communities.png` (Claim C081)
  - `milestones/M09_behavior_bots/figures/behavior_roc_curves.png`
- **Results & Ledger:**
  - `milestones/M09_behavior_bots/results/M09-bot-detection-s42.json`
  - `milestones/M09_behavior_bots/ledger_delta.csv` (Claims C080, C081)
  - `milestones/test_set_access_log.csv` and `milestones/compute_ledger.csv`
  - `milestones/M09_behavior_bots/AUDIT.md`, `README.md`, `worklog.md`
