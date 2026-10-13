# Pre-Registration: Milestone M11 — Track-Level Multimodal Fusion, Full System, Ablations, Shapley Attribution, and Statistical Significance

**Milestone:** M11 — Track-Level Fusion, Full System, Ablations, Shapley Attribution, Significance  
**Repository:** `gurarpitzz/Ares-Cognitive-Computation`  
**Date:** October 13, 2026  
**Status:** PRE-REGISTERED  
**Commit Baseline:** `cb678de` (`m10-complete`)  
**Hardware Target:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Intel Core i7-13620H, 16 GB RAM  
**Execution Environment:** Windows 11 24H2, Python 3.11.9, PyTorch 2.6.0+cu124, transformers 4.49.0, scikit-learn 1.8.0, scipy 1.17.1  
**Target Ledger Claims:** `C001`, `C043`, `C046`, `C047`, `C048`, `C049`, `C051`, `C052`, `C053`, `C054`, `C055`, `C057`, `C058`, `C065`, `C068`, `C069`, `C097`  

---

## 1. Scientific Objective and Context

Sections XXII–XXV, XXVII, and XXX of the ARES monograph present the core system evaluation, claiming a monolithic overall F1-score of 0.92 across multimodal misinformation detection (Claim `C001`), extensive unimodal and layer ablations (Claims `C043`, `C046`, `C047`, `C048`, `C049`), super-additive second-order interactions (Claim `C055`), and uniform statistical significance ($p < 0.01$ via $K=5$ runs of Wilcoxon signed-rank test; Claim `C051`).

The comprehensive project audit (`01_MANUSCRIPT_AUDIT.md`) uncovered critical methodological and mathematical defects in these claims:
1. **Monolithic F1 Impossibility (`C001`):** No single public dataset provides all modalities (text, audio, video, network graph, temporal drift, geopolitical metadata) for the same items. Reporting a single F1 = 0.92 conflated heterogeneous tasks into an ungrounded headline number.
2. **Wilcoxon $K=5$ Mathematical Impossibility (Audit Issue M-1, Claim `C051`):** For a two-sided Wilcoxon signed-rank test with sample size $K = 5$, the minimum possible $p$-value is $2 \times (1/2)^5 = 0.0625$. The manuscript's claim of $p < 0.01$ across all benchmarks with $K=5$ runs is mathematically impossible.
3. **Second-Order Interaction Sign Contradiction (Audit Issue I-5, Claim `C055`):** The manuscript reported $\Delta_{\text{MM}} = 0.071$, $\Delta_{\text{Drift}} = 0.049$, and claimed the joint ablation drop $\Delta_{\text{MM+Drift}} = 0.112 \pm 0.009$ was "super-additive". However, $0.071 + 0.049 = 0.120$. Because $0.112 < 0.120$, the joint effect is strictly **sub-additive** by the manuscript's own arithmetic.
4. **Architectural & Loss Space Mismatch (Audit Issue I-11, Claim `C068`):** The manuscript claimed a 3-way softmax output space ("True / False / Uncertain") while training on binary datasets. In ARES v2, this is resolved rigorously via decision-theoretic Chow cost-asymmetric abstention (Theorem C, M05).
5. **Arbitrary Attribution Weights (Claim `C058`):** The reported normalised weights ($\alpha_{\text{drift}} \approx 0.21, \alpha_{\text{MM}} \approx 0.30$) were derived heuristically. In ARES v2, they are replaced with exact cooperative game-theoretic Shapley values satisfying the efficiency axiom.

Milestone M11 replaces all legacy claims with honest, empirical evaluations across the pre-registered ARES-Bench tracks (T1–T9), executing out-of-fold late fusion, end-to-end composite loss training (Appendix A + Eq. 14), retrained leave-one-out ablations, exact Shapley attribution, and rigorous statistical hypothesis testing with $K=10$ seeds.

---

## 2. Pre-Registered ARES-Bench Tracks & Datasets

To prevent post-hoc track selection (Audit Red Flag RF2), the tracks entering the headline evaluation are pre-registered unconditionally:

| Track ID | Track Name | Datasets | Active Modules | Primary Metric |
|---|---|---|---|---|
| **T1** | Text Veracity | Constraint-COVID, CoAID, LIAR, X-FACT | Text, Logic, Temporal, Geo | Macro-F1 |
| **T2** | Evidence-Based Verification | FEVER 1.0 (with Wiki corpus) | Evidence ($R, S, N, W$), Logic, Text, Temporal, Geo | Macro-F1 & Label Acc |
| **T3** | Image–Text Misinformation | Fakeddit (multimodal subset) | Text, Vision, Metadata | Macro-F1 |
| **T4** | Video Deepfake | Celeb-DF v2 / FF++ (face crops) | Vision (ResNet-50 / Xception) | Video-level AUC & Macro-F1 |
| **T5** | Audio Spoof / Voice | ASVspoof 2019 LA | Audio (Declared CNN) | EER & Macro-F1 |
| **T6** | Audio-Visual Deepfake | Multimodal AV Benchmark | Vision, Audio, AV-sync | AUC & Macro-F1 |
| **T7** | Social Propagation | UPFD (PolitiFact / GossipCop) / PHEME | Text, Behavior, Temporal, Geo | Macro-F1 |
| **T8** | Bot Detection | TwiBot-20 / Cresci-2017 | Behavior (Stylometry, Temporal, Profile, Graph) | Macro-F1 & AUC |
| **T9** | Claim Mutation / Matching | PAWS, STS-B | Temporal / Drift (Lipschitz, Semantic) | ROC-AUC & Macro-F1 |

### Headline Aggregation Rule (Replacing Claim `C001`)
The headline metric replacing `C001` is:
1. The per-track primary metric table reporting Full ARES, best single module, and best baseline with 95% bootstrap confidence intervals.
2. The unweighted arithmetic mean of macro-F1 over the pre-registered classification tracks (T1–T9):
   $$\bar{F}_{1,\text{macro}} = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} F_{1,\text{macro}}^{(t)}$$
No tracks may be added, dropped, or re-weighted after observing test results.

---

## 3. Fusion Architecture Specifications

### 3.1 Late Fusion Heads (M11.2)
1. **Calibrated LLR Logistic Regression:** Evaluates calibrated log-odds from active modules:
   $$\log \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = w_0 + \sum_{m=1}^{M} w_m \ell_m(x_m)$$
   where $\ell_m(x_m)$ are Platt-calibrated log-likelihood ratios from upstream modules (M03–M10). Out-of-fold training ensures no data leakage (Audit Red Flag RF1).
2. **Gradient-Boosted Decision Trees:** Non-linear tree ensemble with early stopping on validation split.
3. **Appendix-A Multilayer Perceptron (MLP):** Input vector of concatenated module scores/embeddings $\to$ Linear(1024) $\to$ GELU $\to$ Dropout(0.1) $\to$ Linear(512) $\to$ GELU $\to$ Dropout(0.1) $\to$ Linear(C).
4. **Chow Cost-Asymmetric Abstention (Resolving Audit Issue I-11, Claim `C068`):**
   Thresholds $\tau_0 = \ln \frac{c_a}{c_{\text{fn}} - c_a}$, $\tau_1 = \ln \frac{c_{\text{fp}} - c_a}{c_a}$ with $c_{\text{fp}} = 1.0, c_{\text{fn}} = 1.0, c_a = 0.20$. Samples in $[\tau_0, \tau_1]$ output `"Uncertain"` (simulated analyst escalation).

### 3.2 End-to-End Fusion Head (M11.3, Appendix A + Eq. 14)
Where modalities co-occur (T3 Image–Text and T6 Audio–Visual):
- **Encoders:** DeBERTa-v3-base text encoder + ResNet-50 vision encoder.
- **Fusion Layer:** Concatenated representations $\to$ 1024-d projection MLP.
- **Composite Loss Function (Eq. 14, Claims `C068`, `C069`):**
  $$\mathcal{L}(\theta) = \lambda_{\text{cls}} \mathcal{L}_{\text{CE,weighted}}(f_\theta(\mathbf{x}), y) + \lambda_{\text{adv}} D_{\text{KL}}(f_\theta(\mathbf{x}) \parallel f_\theta(\mathbf{x} + \boldsymbol{\delta})) + \lambda_{\text{reg}} \|\theta\|_2^2$$
  - Pre-registered hyperparameters: $\lambda_{\text{cls}} = 1.0$, $\lambda_{\text{adv}} = 0.3$, $\lambda_{\text{reg}} = 10^{-4}$.
  - Class weights: $(0.7, 1.3)$ for classes $(0, 1)$ reflecting label imbalance.
  - Virtual Adversarial Perturbation: $\boldsymbol{\delta} = \varepsilon \frac{\mathbf{g}}{\|\mathbf{g}\|_2}$ where $\mathbf{g} = \nabla_{\mathbf{d}} D_{\text{KL}}(f(\mathbf{e}) \parallel f(\mathbf{e} + \xi \mathbf{d}))$, $\mathbf{d} \sim \mathcal{N}(0, I)$, with perturbation radius $\varepsilon = 0.02$, $\xi = 10^{-6}$.
  - Optimizer: AdamW ($lr = 2 \times 10^{-5}$, weight decay $= 0.01$). Notice that weight decay operates alongside $\lambda_{\text{reg}}$ (double regularization as declared in manuscript, flagged per Audit Issue I-16).
- **Training Mechanism Ablations:**
  1. Class weighting ablation: Set weights to $(1.0, 1.0)$ to empirically measure the false-class recall drop (Claim `C057`).
  2. Adversarial regularization ablation: Set $\lambda_{\text{adv}} = 0$ to quantify robustness contribution.

---

## 4. Ablation & Interaction Protocols

### 4.1 Leave-One-Module-Out Retraining (Audit Red Flag RF3, RF4)
- **Constraint RF3 (Strict Retraining):** Zeroing inputs at test time is strictly forbidden. The fusion head is retrained from scratch on the reduced feature set $\mathcal{M} \setminus \{X\}$.
- **Constraint RF4 (Existence Check):** Ablations are conducted only where module $X$ is active in that track:
  - T1/T2: −Evidence, −Logic, −Temporal, −Geo, Text-only.
  - T3: −Vision, −Text, −Metadata.
  - T6: −Audio, −Vision, −AV-sync.
  - T7: −Behavior, −Temporal, −Geo, Text-only.
  - T8: −Profile, −Temporal, −Graph, −Stylometry.
  - All tracks: −HITL (escalation disabled, forced hard argmax).
- **Metric Drop:** $\Delta \text{Metric} = \text{Metric}(\text{Full}) - \text{Metric}(\text{Full} \setminus \{X\})$, reported with 95% bootstrap confidence interval (B = 2,000 resamples).
- **Conflict Domain Drop (Claim `C054`):** For claims tagged as `Conflict` in M10, evaluate the specific recall drop when omitting the geopolitical layer.

### 4.2 Second-Order Interaction Protocol (Audit Red Flag RF6, Claim `C055`)
- **2×2 Factorials:**
  1. Multimodal $\times$ Temporal on multimodal tracks with temporal dynamics.
  2. Logic $\times$ Geo on evidence tracks (T1/T2).
- **Interaction Statistic:**
  $$\Delta_{\text{interaction}} = \Delta_{A \wedge B} - (\Delta_A + \Delta_B)$$
  where $\Delta_S = \text{Metric}(\text{Full}) - \text{Metric}(\text{Full} \setminus S)$.
- **Classification Rule:**
  - If the 95% bootstrap CI of $\Delta_{\text{interaction}}$ is strictly positive ($[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}] > 0$): **Super-additive**.
  - If the 95% bootstrap CI of $\Delta_{\text{interaction}}$ is strictly negative ($[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}] < 0$): **Sub-additive**.
  - If the 95% bootstrap CI spans zero: **No detectable interaction**.
- Promotional language claiming super-additivity when the empirical interval contains negative values is strictly forbidden.

---

## 5. Exact Cooperative Game-Theoretic Shapley Attribution (Claim `C058`, Audit Red Flag RF5)

To eliminate heuristic attribution weights, module contributions are evaluated via exact Shapley values over all $2^{|\mathcal{M}|}$ coalitions:
$$\phi_i = \sum_{S \subseteq \mathcal{M} \setminus \{i\}} \frac{|S|! (|\mathcal{M}| - |S| - 1)!}{|\mathcal{M}|!} \left[ v(S \cup \{i\}) - v(S) \right]$$
where $v(S)$ is the test macro-F1 of a fusion model trained using only module subset $S$, with $v(\emptyset)$ defined as majority-class baseline performance.

### Verification of Shapley Axioms:
1. **Efficiency Axiom Check (RF5):** $\sum_{i \in \mathcal{M}} \phi_i \equiv v(\mathcal{M}) - v(\emptyset)$ must hold exactly (within numerical precision $< 10^{-6}$).
2. **Symmetry:** If $v(S \cup \{i\}) = v(S \cup \{j\})$ for all $S$, then $\phi_i = \phi_j$.
3. **Dummy Axiom:** If $v(S \cup \{i\}) = v(S)$ for all $S$, then $\phi_i = 0$.
4. **Confidence Intervals:** 95% bootstrap confidence intervals computed over $B = 2,000$ test item resamples.
5. **Normalized Shares:** $\alpha_i = \frac{\phi_i}{\sum_{j \in \mathcal{M}} \phi_j}$.

---

## 6. Statistical Significance Protocol (Audit Issue M-1, Claims `C051`, `C052`)

### 6.1 Mathematical Resolution of Audit M-1 ($K=10$ Seeds)
The manuscript's claim of $p < 0.01$ with $K=5$ runs under a two-sided Wilcoxon signed-rank test is impossible, because:
$$p_{\min}(K=5) = 2 \times \left(\frac{1}{2}\right)^5 = \frac{2}{32} = 0.0625 > 0.01$$
In Milestone M11, we explicitly execute **$K = 10$ independent random seeds** ($s \in \{42, 101, 202, 303, 404, 505, 606, 707, 808, 909\}$), where the theoretical minimum two-sided Wilcoxon $p$-value is:
$$p_{\min}(K=10) = 2 \times \left(\frac{1}{2}\right)^{10} = \frac{2}{1024} = 0.001953 < 0.01$$
This resolution is transparently documented in all summary tables and the monograph audit.

### 6.2 Significance Testing Battery
For each pairwise comparison between ARES and baselines/ablations:
1. **Instance-Level Tests:**
   - Paired percentile bootstrap difference $\Delta$ with 95% CI ($B = 10,000$).
   - Exact McNemar test on discordant error pairs with continuity correction.
2. **Seed-Level Tests ($K = 10$):**
   - Paired Student's $t$-test.
   - Exact two-sided Wilcoxon signed-rank test.
3. **Standardized Effect Size (Claim `C052`):**
   - Cohen's $d_z = \frac{\bar{D}}{s_D}$ where $D_k = \text{Metric}_{\text{ARES}, k} - \text{Metric}_{\text{Base}, k}$, with 95% analytic confidence interval.
4. **Multiple Testing Correction:**
   - Step-down Holm-Bonferroni correction within each track family, enforcing family-wise error rate $\alpha = 0.05$. Exact adjusted $p$-values are reported.

---

## 7. Audit Red Flag Defenses

| Red Flag | Description | Enforcement Mechanism |
|---|---|---|
| **RF1** | Stacking Leakage | Strict out-of-fold training for fusion heads. No train row may use in-fold component predictions. Verified by `is_out_of_fold == True` assertion. |
| **RF2** | Post-Hoc Track Selection | Headline table matches the pre-registered tracks T1–T9 exactly. No tracks added or omitted. |
| **RF3** | Ablation Retraining | Every ablation retrains the fusion model without the excluded module; zeroing inputs at test time is strictly blocked. |
| **RF4** | Module Existence Check | Ablations are reported only where the module's input exists in that benchmark. |
| **RF5** | Shapley Efficiency Axiom | Shapley values sum exactly to $v(\text{Full}) - v(\emptyset)$, verified by unit tests. |
| **RF6** | Interaction Language | Interactions are termed super-additive or sub-additive strictly based on bootstrap CIs. |
| **RF7** | Exact $p$-Values | All $p$-values reported with exact numbers, citing the Holm-Bonferroni correction. |
| **RF8** | SOTA Plausibility | Track metrics are benchmarked against M02 literature baselines; anomalies are investigated. |

---

## 8. Target Deliverables & Artifacts

1. `milestones/M11_fusion_ablation/PREREG.md`: This document.
2. `ares_v2/src/ares/fusion/`:
   - `assemble.py`: Multi-track feature assembly with out-of-fold partitioning.
   - `late.py`: Calibrated LLR, gradient-boosted trees, and 1024-d MLP with Chow abstention.
   - `e2e.py`: Appendix A + Eq. 14 composite loss with embedding-space VAT.
   - `ablation.py`: Leave-one-out retraining harness and interaction tester.
   - `shapley.py`: Exact Shapley attribution engine with efficiency validation.
   - `significance.py`: Paired bootstrap, McNemar, paired t, Wilcoxon ($K=10$), Cohen's $d_z$, Holm-Bonferroni.
3. `ares_v2/tests/test_fusion.py`: Comprehensive test suite verifying all mathematical and architectural invariants.
4. `ares_v2/scripts/run_m11_evaluation.py`: Master reproducible evaluation pipeline.
5. Publication tables (`milestones/M11_fusion_ablation/tables/`):
   - `table1_headline_tracks.csv` (replaces `C001`)
   - `table23_ablation_summary.csv` (replaces `C043`)
   - `table27_modality_removal.csv` (replaces `C046`)
   - `table28_geo_ablation.csv` (replaces `C047`, `C054`)
   - `table29_behavior_ablation.csv` (replaces `C048`)
   - `table30_temporal_ablation.csv` (replaces `C049`)
   - `table34_significance_tests.csv` (replaces `C051`, `C052`)
   - `table35_training_mechanism_ablation.csv` (replaces `C057`)
   - `shapley_attribution.csv` (replaces `C058`)
   - `second_order_interactions.csv` (replaces `C055`)
   - `baseline_comparison.csv` (replaces `C065`, `C068`)
6. Publication figures (`milestones/M11_fusion_ablation/figures/`):
   - `fig39_shapley_attribution.png`
   - `fig40_ablation_waterfall.png`
   - `fig41_baseline_progression.png`
   - `fig42_interaction_effects.png`
7. Schema-compliant result JSON: `milestones/M11_fusion_ablation/results/M11-fusion-ablation-s42.json`.
8. Auditing artifacts: `milestones/M11_fusion_ablation/AUDIT.md`, `README.md`, `worklog.md`, `ledger_delta.csv`.
9. Tags: `m11-complete` and model freeze tag `ares-v2-freeze-1`.
