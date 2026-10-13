# M09 — Behavioral & Bot Detection, Propagation Modelling

**Objective:** Implement Section VII exactly as specified: stylometry (char n-grams, punctuation variance, Yule's K) + One-Class SVM, inter-post-interval HMM log-likelihood ratio Λ, centrality/graph features. Evaluate on bot benchmarks (T8) and as behavioral signals for social-propagation veracity (T7). Replace the illustrative Figs. 10–12 with real data.
**Ledger IDs:** C080, C081 (+ behavior inputs for C043, C048, C053 in M11).
**Depends on:** M02.
**Compute:** RTX 4060 Laptop, ≈ 5 GPU-h for the GNN baselines (BotRGCN on TwiBot-20 fits in 8 GB; use neighbour sampling if needed). Feature engineering runs on the CPU with streamed JSON/parquet (16 GB RAM).
**Output dir:** `milestones/M09_behavior_bots/`

---

## Pre-registration
- **T8 bot detection:** Cresci-2017 (grouped split by account) and TwiBot-20 (official split). TwiBot-22 only as a documented subgraph, if storage allows; otherwise state it was excluded on storage grounds. **Tier 2 (cross-dataset):** train on TwiBot-20 or Cresci-2017, test on Cresci-2015 and the Botometer repository sets, using profile features only. Metrics: F1 (bot = positive), accuracy, AUC.
- **T7 propagation veracity:** PHEME (leave-one-event-out) and UPFD-PolitiFact/GossipCop (official splits). Metric: macro-F1.
- **H9.1:** Each behavioral family (stylometry, temporal-HMM, graph, profile) beats the majority baseline on T8 (paired bootstrap, Holm).
- **H9.2:** The OCSVM authorship-anomaly detector (trained on human accounts only) has AUC > 0.5 for bot detection (descriptive).
- **H9.3:** Behavioral/propagation features add to text-only veracity on T7 (tested formally in M11; measured standalone here).

## Steps

### M09.1 — Stylometry features (Tab. VII exactly)
- **Action:** `src/ares/behavior/stylometry.py`: char n-gram frequencies n = 2, 3, 4 (hashed / top-K vocabulary from train only), punctuation variance σ² = (1/N)Σ(pᵢ − μ_p)² (define pᵢ precisely, e.g. punctuation rate per post), Yule's K = 10⁴·(Σᵢ i²·f_i − N)/N² (define f_i as the number of word types occurring i times; N tokens). Unit tests against hand-computed examples.

### M09.2 — One-Class SVM authorship anomaly (Sec. VII-A)
- **Action:** Train OCSVM (RBF; ν, γ chosen on val by AUC) on human accounts' stylometric vectors from train; score all test accounts; AUC for bot detection.

### M09.3 — Temporal posting patterns & HMM LLR (Sec. VII-B)
- **Action:** `behavior/temporal.py`: inter-post intervals Δtᵢ; entropy of binned intervals; periodicity (autocorrelation peak / spectral peak). HMMs (`hmmlearn`, 2–4 hidden states, Gaussian on log-intervals) fitted separately on bot and human **train** accounts; Λ = log P(O | λ_bot) − log P(O | λ_human) per test account. State count chosen on val (BIC + val AUC). Only posts **before** a cutoff are used for features if any temporal evaluation is involved.

### M09.4 — Graph features (Sec. VII-C)
- **Action:** `behavior/graph.py`: eigenvector centrality (Eq. in VII-C), PageRank, degree, clustering, community (Louvain) features on the follow/retweet graph (TwiBot); label propagation baseline; plus published GNN baselines from the TwiBot benchmark repo (e.g. BotRGCN) for reference.

### M09.5 — Profile features
- **Action:** Standard metadata features (followers/friends ratio, account age, statuses count, default profile, etc.), as in Botometer-style feature sets.

### M09.6 — T8 bot detection evaluation
- **Action:** Classifier per family + combined (GBM/logistic), K = 10 seeds where stochastic. Tables with CIs; comparison to published TwiBot-20 / Cresci-2017 numbers from `sota_reference.csv`.

### M09.7 — T7 propagation veracity
- **Action:** UPFD: run the PyG UPFD/GCNFN baselines (official example code) with fixed seeds; PHEME LOEO: features from reply-tree structure (depth, breadth, reply latency distribution), user features, temporal features (time-to-first-reply, burstiness), and transferred bot scores (model trained on Cresci/TwiBot profile features applied to PHEME users. **Flag this as domain transfer.**). Cache per-item behavioral scores out-of-fold for M11.

### M09.8 — Real figures (C080, C081)
- **Action:** Fig. 11 replacement: posting-interval histograms/timelines for real bot vs human accounts (Cresci), with account IDs anonymised. Fig. 12 replacement: a real TwiBot subgraph with detected communities, coloured by label and predicted score. Fig. 10 pipeline diagram redrawn cleanly.

---

## Audit gate — milestone-specific red flags
- RF1: Account-level splits (no user in two splits); n-gram vocabulary and scalers fitted on train only.
- RF2: Temporal features use no posts after the evaluation cutoff.
- RF3: PHEME LOEO folds hold out entire events.
- RF4: Known TwiBot-20 benchmark caveats checked against the benchmark paper; results compared with published baselines.
- RF5: Bot-score transfer to PHEME labelled as transfer; its validity limitations written up.
- RF6: Anonymisation in figures (no handles of real private individuals).

## Exit criteria
- [ ] Section VII components implemented exactly and tested.
- [ ] T8 and T7 evaluations with CIs; caches for fusion.
- [ ] Real Figs. 10–12; ledger C080, C081 updated.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m09-complete`.
