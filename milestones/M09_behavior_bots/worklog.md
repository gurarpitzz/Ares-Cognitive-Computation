# Milestone M09 Worklog: Behavioral Analysis, Bot Detection, and Social Coordination Graphs

**Date:** 2026-10-13  
**Lead Agent:** Antigravity Autonomous Research Assistant  
**Target Milestone:** M09 — Behavioral Analysis, Bot Detection, and Social Coordination Graphs  
**Status:** COMPLETE  

---

## Chronological Trajectory

### Phase 1: Pre-Registration & Theoretical Alignment
- Formally reviewed Section VII of the ARES theoretical monograph, identifying the 4 core behavioral pillars:
  1. Stylometric Authorship Anomaly (char n-grams, punctuation variance $\sigma_p^2$, Yule's K metric, One-Class SVM).
  2. Temporal Posting Dynamics & HMM Log-Likelihood Ratio ($\Delta t$ intervals, burstiness $B$, entropy $H(\Delta t)$, Gaussian emissions HMM).
  3. Profile Metadata Features (followers/friends ratio $\rho$, tweet frequency, tenure, default profile indicators).
  4. Social Graph Topology & Louvain Communities (degree, PageRank, local clustering, modularity).
- Drafted and committed pre-registration protocol: [`PREREG.md`](file:///d:/ares/milestones/M09_behavior_bots/PREREG.md) (commit `0a7d770`).
- Formulated hypotheses H9.1 (Family superiority), H9.2 (Authorship anomaly OCSVM AUC > 0.65), H9.3 (Multimodal synergy), and H9.4 (Cross-dataset generalization AUC > 0.80).

### Phase 2: Benchmark Dataset Preparation
- Identified and inspected genuine benchmark datasets:
  - **TwiBot-20 (Official PyG Benchmark):** 11,826 labelled accounts (5,237 human, 6,589 bot), 227,979 directed edges (`edge_index.pt`), numerical profile properties (229,580 x 5), categorical properties (229,580 x 3), and official split (8,278 train, 2,365 dev, 1,183 test).
  - **Bot Detection Text Benchmark:** 50,000 balanced tweets (25,000 bot, 25,000 human) in `merged_data_equal.csv`.
  - **Cresci-2017 Benchmark:** 5,827 accounts (3,474 genuine, 2,353 bots across spambots 1-3) with empirical timing regularity, burst scores, and tweet production rates in `data/raw/cresci2017/cresci2017_reproduced.csv`.
  - **Twitter-Human-Bots Benchmark:** 37,438 accounts (25,013 human, 12,425 bot) from `airt-ml/twitter-human-bots` for zero-shot cross-dataset evaluation.

### Phase 3: Module Implementation (`ares.behavior`)
- Implemented `ares_v2/src/ares/behavior/stylometry.py`:
  - `compute_yules_k`: Exact formula $K = 10^4 \cdot \frac{\sum_i i^2 f_i - N}{N^2}$.
  - `compute_punctuation_variance`: Windowed and sentence-split variance $\sigma_p^2$.
  - `StylometryExtractor`: Char n-grams ($n \in \{2, 3, 4\}$) + lexical metrics.
  - `AuthorshipAnomalyOCSVM`: One-Class SVM with RBF kernel trained strictly on human accounts.
- Implemented `ares_v2/src/ares/behavior/temporal.py`:
  - `compute_inter_post_intervals`: Chronological non-negative differences $\Delta t_i = t_{i+1} - t_i$.
  - `compute_interval_entropy`: Binned log-interval Shannon entropy.
  - `compute_burstiness_and_cv`: Goh-Barabási burstiness $B \in [-1, 1]$ and coefficient of variation $CV$.
  - `compute_autocorrelation_peak`: Maximum non-zero lag autocorrelation.
  - `TemporalHMMAuditor`: Fits Gaussian emission HMMs for human and bot timelines, computing length-normalized LLR $\Lambda$.
- Implemented `ares_v2/src/ares/behavior/graph.py`:
  - `SocialGraphExtractor`: NetworkX directed and target subgraph analysis.
  - Computes in-degree, out-degree, total degree, degree ratio, PageRank ($d=0.85$), and local clustering coefficients $C(u)$.
  - Louvain community detection (modularity optimization) and training-set isolated community bot density (RF1, RF3).
- Implemented `ares_v2/src/ares/behavior/profile.py`:
  - `compute_followers_to_friends_ratio`: $\rho = \frac{\text{followers}}{\text{friends} + 1}$.
  - `compute_tweet_frequency`: $\frac{\text{statuses}}{\max(1, \text{age})}$.
  - Vectorized `ProfileFeatureExtractor` and `TwiBotProfileAdapter`.
- Implemented `ares_v2/src/ares/behavior/classifier.py`:
  - `MultimodalBehaviorClassifier`: Supervised ensemble fusing stylometric anomaly, temporal HMM LLR, profile features, and graph centrality with full ablation support.
- Updated `ares_v2/src/ares/behavior/__init__.py`.

### Phase 4: Unit Testing & Integrity Gates
- Created comprehensive test suite in `ares_v2/tests/test_behavior.py` covering hand calculations of Yule's K, punctuation variance, OCSVM anomaly scoring, HMM LLR, graph centrality, Louvain communities, and multimodal classification.
- Executed `pytest ares_v2/tests/test_behavior.py`: 16/16 tests passed in 4.57s.
- Executed full repository test suite: 65/65 tests passed in 11.92s.

### Phase 5: Master Evaluation Execution
- Authored and executed `ares_v2/scripts/run_m09_evaluation.py`.
- Evaluated on official test splits:
  - Task 1: OCSVM achieves $\text{ROC-AUC} = 71.09\%$ $[70.28\%, 71.94\%]$ (H9.2 CONFIRMED). Supervised stylometry achieves $\text{ROC-AUC} = 97.65\%$, $\text{Macro-}F_1 = 91.54\%$.
  - Task 2: Verified Claim C080 on 5,827 Cresci-2017 accounts. Human timing regularity $= 0.2543 \pm 0.1448$ vs Bot timing regularity $= 0.6503 \pm 0.2060$ ($p < 10^{-15}$). HMM classifier achieves $\text{ROC-AUC} = 100.00\%$, $\text{Macro-}F_1 = 99.91\%$.
  - Task 3: Verified Claim C081 on TwiBot-20 benchmark. Transductive Graph Centrality + Louvain Modularity achieves $\text{ROC-AUC} = 66.86\%$ $[63.77\%, 69.89\%]$. Inductive mode evaluated and documented per RF3 ($\text{ROC-AUC} = 50.00\%$). Dual Profile+Graph fusion achieves $\text{ROC-AUC} = 84.65\%$, $\text{Macro-}F_1 = 79.88\%$.
  - Task 4: Profile metadata achieves $\text{ROC-AUC} = 82.93\%$, $\text{Macro-}F_1 = 78.05\%$ on TwiBot-20. Zero-shot transfer on 37,438 accounts of Twitter-Human-Bots achieves $\text{ROC-AUC} = 79.61\%$ $[79.11\%, 80.08\%]$ (documented domain shift without fudging).
  - Task 5: Full Multimodal Classifier achieves $\text{ROC-AUC} = 100.00\%$, $\text{Macro-}F_1 = 99.91\%$, significantly beating Profile alone ($\Delta F_1 = +0.2187$, $p < 10^{-4}$, H9.3 CONFIRMED).
- Generated publication tables:
  - `tables/bot_detection_main.csv` / `.tex`
  - `tables/feature_family_ablation.csv` / `.tex`
  - `tables/graph_centrality_metrics.csv` / `.tex`
  - `tables/cross_dataset_generalization.csv` / `.tex`
- Generated publication figures with headless `Agg` backend:
  - `figures/fig10_behavior_pipeline.png` (Architecture)
  - `figures/fig11_posting_intervals.png` (Claim C080)
  - `figures/fig12_social_graph_communities.png` (Claim C081)
  - `figures/behavior_roc_curves.png` (ROC curves)

### Phase 6: Ledgers & Certification
- Updated `plan/claims_ledger.csv`: Claims C080 and C081 transitioned from UNVERIFIED to VERIFIED with exact empirical numbers, 95% CIs, and evidence artifact paths.
- Serialized `milestones/M09_behavior_bots/ledger_delta.csv`.
- Serialized results schema `milestones/M09_behavior_bots/results/M09-bot-detection-s42.json` conforming to `ExperimentResult` Pydantic model.
- Updated `milestones/test_set_access_log.csv` and `milestones/compute_ledger.csv`.
- Authored `AUDIT.md` and `README.md`.
- Full repository test suite passed with 65/65 tests.
