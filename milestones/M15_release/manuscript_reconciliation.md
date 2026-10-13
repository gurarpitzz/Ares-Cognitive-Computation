# ARES 2.0 Manuscript Reconciliation & Rewrite Protocol

**Document Version:** 2.0.0  
**Milestone:** M15 — Release, Reproducibility Package, References & Manuscript Reconciliation  
**Target Manuscript:** "ARES: Autonomous Real-Time Evaluation System for Multimodal Misinformation Detection" (Cognitive Computation Submission)  
**Auditor Protocol:** Every number appearing in this reconciliation is strictly grounded in `plan/claims_ledger.csv` with status `REPRODUCED`, `REVISED`, or `VERIFIED` and an existing empirical evidence path. Claims marked `REMOVE` are excised.

---

## 1. Title Revisions (Addressing Reviewer 2)

**Critique (Reviewer 2):** *"The paper title is overly general and fails to clearly reflect the core technical contribution or methodological innovation of this study."*

**Recommended Title Options:**
1. **(Recommended):** *ARES: Evidence-Grounded, Calibrated Multimodal Misinformation Verification with Mutation-Aware Claim Memory and Decision-Theoretic Abstention*
2. *Calibrated Multimodal Misinformation Detection: Unifying Sparse-Dense Evidence Retrieval, Cross-Modal Verification, and Semantic Mutation Memory*
3. *A Decision-Theoretic Framework for Multimodal Fact-Checking: Integrating Dense Evidence Retrieval, Deepfake Forensics, and Calibrated Bayesian Fusion*

---

## 2. Abstract & Conclusion Overhaul (Addressing Reviewers 1 and 3)

**Critique (Reviewer 3):** *"It claims high performance (F1-score: 0.92) and low latency (415ms/claim) but provides limited empirical support... promotional language in the conclusion."*  
**Action:** Replace ungrounded monolithic F1 (0.92) and ungrounded latency (415 ms) with scope-demarcated empirical measurements and per-track results across declared benchmarks. Excise all promotional adjectives ("revolutionary", "unprecedented", "flawless", "superior").

### Exact Revised Abstract Text:
> Real-time mitigation of digital misinformation requires verifying claims across heterogeneous, non-co-occurring modalities while accounting for semantic narrative mutation, geopolitical context, and verification uncertainty. We present ARES 2.0, a modular, open-source platform that unifies sparse/dense evidence retrieval, cross-modal neural verifiers (text, vision, audio), semantic mutation memory, and decision-theoretic abstention. Rather than asserting a monolithic metric across artificial multi-modal combinations, we evaluate ARES across established benchmark tracks: text veracity on LIAR, Constraint-COVID, and CoAID (Macro-F1 = 0.901 [95% CI: 0.892, 0.910]); dense evidence retrieval on FEVER 1.0 (Label Accuracy = 0.842 [0.831, 0.853]); facial deepfake detection on FaceForensics++ c23 (Video-level AUC = 0.952 [0.944, 0.960]); and speech anti-spoofing on ASVspoof 2019 (Macro-F1 = 0.976 [0.969, 0.983]). By implementing cost-sensitive Chow abstention over temperature-calibrated posterior probabilities (ECE = 0.0312), the decision layer achieves 87.3% selective accuracy under a 6.8% escalation rate. On an edge-tier laptop GPU (NVIDIA RTX 4060, 8 GB VRAM), median per-claim verification latencies span 14.8 ms (text-alone), 23.8 ms (dense retrieval), 27.2 ms (16-frame video), and 1.3 ms (speech spoofing), with sustained service throughput of 3,560 req/s. All code, model weights, evaluation harnesses, and cryptographic audit logging tools are openly released under Apache-2.0.

### Exact Revised Conclusion Text:
> We have presented ARES 2.0, an evidence-grounded framework for multimodal misinformation verification that prioritizes mathematical rigor, calibrated uncertainty, and operational feasibility. Our empirical evaluation across seven benchmark tracks demonstrates that integrating domain-specific evidence retrieval, semantic drift monitoring, and Chow abstention improves veracity assessment relative to isolated modal baselines. Importantly, our systems evaluation refutes prior theoretical assumptions of quadratic transformer scaling at operational token lengths ($n \le 512$, empirical slope $\alpha = 0.040$), demonstrating that consumer-grade edge hardware (8 GB VRAM) is sufficient for real-time inference without enterprise server clusters. Major open challenges remain in mitigating cross-domain performance degradation (average macro-F1 drop of $\Delta = -0.191$ under out-of-domain distribution shift) and scaling dense retrieval over non-English corpora. The complete, verified open-source release provides a reproducible foundation for future research in trustworthy multi-modal content verification.

---

## 3. Section XIII: Real-World Use Cases & Impact Claims (Claims C024, C025, C026, C027, C030, C031)

**Action:** **REMOVE / REVISE.** All claims asserting unverified real-world deployment impacts, government agency integrations, classified military interventions, autonomous social media post takedowns, and unapproved human subject surveys are excised from the manuscript.

### Detailed Subsection Rewrite Instructions:
1. **Section XIII-A (Assam Flood Disinformation, Claim C024):**
   - **REMOVE:** *"Level-3 alert issued within 18 minutes; panic traffic decreased 47%; heatmap integrated into national disaster management dashboard."*
   - **REPLACE WITH:** A controlled evaluation on **ARES-Synth v1** (`Assam_Flood_Scenario`), explicitly labeled **[SIMULATED BENCHMARK EVALUATION]**. Report empirical macro-F1 of 0.8989 under synthetic flood crisis claims, documenting that real-world disaster traffic data was not collected.
2. **Section XIII-B (Ladakh Border Propaganda, Claim C025):**
   - **REMOVE:** *"Classified alert issued to military intelligence in Ladakh within 12 minutes, preventing military escalation."*
   - **REPLACE WITH:** A controlled evaluation on **ARES-Synth v1** (`Ladakh_Border_Scenario`), explicitly labeled **[SIMULATED BENCHMARK EVALUATION]**. Document geopolitical NER and GDELT event corroboration performance on synthetic border tension narratives without claiming operational military alerting.
3. **Section XIII-C (Kashmir Lockdown Media Verification, Claim C026):**
   - **REMOVE:** *"PSA launched 6 hours earlier than mainstream media response."*
   - **REVISE TO:** An empirical retrospective offline verification trace on historical news claims (Table `table_retrospective_traces.csv`). Report calibrated posterior probability $P(Y=\text{false} \mid x) = 0.941$ and local retrieval latency of 23.8 ms.
4. **Section XIII-D (Social Bot Disinformation Swarms, Claim C027):**
   - **REMOVE:** *"13,000+ bot-amplified posts removed across Twitter and Facebook."*
   - **REPLACE WITH:** Offline bot and social spam detection evaluation on peer-reviewed benchmarks (TwiBot-20, Cresci-2017) from Milestone M09. Report empirical macro-F1 of 0.841 and ROC-AUC of 0.9998.
5. **Section XIII-E (Public Perception Survey, Claim C030):**
   - **REMOVE:** *"+62% trust gain vs traditional fact-checks based on a survey of 4,000 respondents."*
   - **REPLACE WITH:** A transparent limitation statement acknowledging that evaluating human trust dynamics requires Institutional Review Board (IRB) approved user studies, which remain designated for future work.
6. **Figure 22 & Section XIII (Analyst Response Latency, Claim C031):**
   - **REMOVE:** Figure 22 claiming empirical analyst response times (~5–30 min).
   - **REPLACE WITH:** Figure 22 as a parameterized theoretical $M/M/c$ queueing sensitivity analysis (`fig22_queueing_sensitivity_simulation.png`), explicitly labeled **[SIMULATED QUEUE SENSITIVITY ANALYSIS]**, exploring queue backlog across hypothetical review rates $\mu \in \{10, 15, 20\}$ claims/hour.

---

## 4. Narrative Sections & "Story Behind ARES" (Addressing Reviewer 3)

**Critique (Reviewer 3):** *"The work includes narrative sections (e.g., 'The Story Behind ARES'). I believe the writing style better suited to blogs than journals."*  
**Action:** **REMOVE** the entire narrative section "The Story Behind ARES". Condense the core motivation into a single, rigorous paragraph within Section I (Introduction) emphasizing the technical necessity of unified evidence-grounded verification.

---

## 5. Dataset Labeling Pipeline (Section XII-B, Claim C090)

**Critique (Reviewer 3):** *"Training details are superficial... Five-stage labeling incl. MTurk + IFCN crowdsourced verification."*  
**Action:** **REVISE.** Clarify that the authors did not commission an independent Amazon Mechanical Turk or IFCN labeling consortium for this paper. Instead, state clearly:
> "ARES 2.0 is evaluated on established, publicly available peer-reviewed benchmarks whose labels were verified by professional fact-checkers (PolitiFact, Snopes, IFCN signatories) or curated academic protocols (LIAR, Constraint, CoAID, X-FACT, FEVER, AVeriTeC). For stress evaluations under domain shift, we introduce ARES-Synth v1, a grounded synthetic benchmark derived from frozen Wikipedia and ReliefWeb sources with code-determined perturbation ground truth."

---

## 6. Mathematical Foundations & Theoretical Results (Section XXI, Claims C070–C075)

**Action:** Replace flawed heuristic derivations with verified theorems from Milestone M01:
1. **Semantic Mutation Drift Theorem (Theorem 1, Claims C070, C071):**
   - Replace heuristic Taylor expansion with the exact non-central $t$-distribution theorem:
     $$\cos(x, x+\epsilon) = \frac{T}{\sqrt{T^2 + d - 1}}, \quad T \sim \text{noncentral-}t\left(\nu = d-1, \, \lambda = \frac{\|x\|}{\sigma}\right)$$
   - State the two-sided Neyman-Pearson detection guarantee bounding both false-alarm and miss probabilities.
2. **Transformer FLOP Scaling & Complexity (Claims C041, C075):**
   - Refute the monograph's claim of quadratic scaling ($\alpha = 1.99$) for $n \le 512$.
   - Present analytical FLOP breakdown: $\text{FLOPs} = 12 \cdot (24nd^2 + 4n^2d)$. Prove that the linear projection term ($24nd^2$) accounts for 90.0% of operations at $n=512$, with quadratic attention crossing over to dominance only at $n^* = 6d = 4,608$ tokens.
   - Report empirical scaling slope measured on DeBERTa-v3: $\alpha = 0.040$ [95% CI: -0.044, 0.127].
3. **Decision Theory & Chow Abstention (Claims C010, C062):**
   - Present Theorem C (Chow cost-sensitive abstention): optimal reject region $\{x : \max_y P(Y=y \mid x) < 1 - c_r / c_s\}$.
   - Correct the expected latency unit error: $E[T] = (1 - r_{\text{esc}})T_{\text{auto}} + r_{\text{esc}}T_{\text{human}} = 75.91\text{ s}$ (with $T_{\text{auto}} = 0.024\text{ s}, T_{\text{human}} = 1116.0\text{ s}, r_{\text{esc}} = 0.068$).

---

## 7. Systems Performance & Hardware Profile (Section XV, Claims C002, C004, C033)

**Action:** Update all systems performance claims from ungrounded enterprise server estimates to verified measurements on the declared laptop platform:
1. **Disaggregated Latency Scopes (Claim C002):** Replace monolithic 415 ms with:
   - Scope A (Text-alone): $p_{50} = 14.8\text{ ms}$
   - Scope B (Dense local retrieval): $p_{50} = 23.8\text{ ms}$
   - Scope C (Visual deepfake): $p_{50} = 27.2\text{ ms}$
   - Scope D (Audio spoofing): $p_{50} = 1.3\text{ ms}$
   - Scope E (Detection-to-alert): $p_{50} = 25.0\text{ ms}$
2. **Throughput & Concurrency (Claim C004):** Replace "~4000 req/s @ C=5000" with empirical saturation curve peaking at **3,560.6 req/s** at concurrency $C=500$ on the RTX 4060 laptop.
3. **Hardware Requirements (Claim C033):** Replace "Min 32 GB RAM, A100 GPU" with:
   > "Evaluated on a consumer-tier laptop: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Intel Core i7-13620H, 16 GB DDR5 RAM. Steady-state peak VRAM consumption is 731.6 MB, confirming feasibility for edge deployment."

---

## 8. Section XIV: Ethics Safeguards Matrix (Claim C084)

**Action:** Reorganize Section XIV into an explicit table distinguishing implemented software from governance proposals:
- **Implemented:** SHA-256 Merkle-style hash-chained audit log (`auditlog.py`, 100% tamper detection rate); automated PII sanitization engine (`pii.py`, 100% recall, zero raw leaks); public rate-limited decision transparency API (`GET /transparency/decisions`); emergency analyst kill switch (`POST /admin/killswitch`).
- **Proposed (Future Work):** Federated learning across regional edge nodes (theoretical architecture proposal); independent civil society oversight board (governance recommendation); ethical data licensing framework (policy proposal).

---

## 9. Bibliographic Overhaul & Reference List (Claim C092)

**Critique (Reviewers 2 and 3):** Missing citations, unresolved placeholders `[?]`, and erroneous authors.  
**Action:** Replace legacy bibliography with the verified 65-entry BibTeX database (`references.bib`). Resolve every inline `[?]` citation:
- [1] $\to$ Zhou & Zafarani (ACM Computing Surveys 2020)
- [4] $\to$ Helbing et al. (Springer 2019)
- [6] $\to$ Graves, Mohamed & Hinton (IEEE ICASSP 2013)
- [8] $\to$ He et al. ResNet (IEEE CVPR 2016)
- [9] $\to$ Qian et al. F3-Net (ECCV 2020)
- [13] $\to$ Grinberg et al. (Science 2019)
- [14] $\to$ Yang et al. (NAACL 2016)
- [16] $\to$ Todisco et al. ASVspoof 2019 (Interspeech 2019)
- [22] $\to$ Durmus et al. FEQA (ACL 2020)
- [26] $\to$ Gordon Pennycook & David Rand (PNAS 2019)
- [27] $\to$ Karimi et al. (ACM CIKM 2018)
- [35] $\to$ Tschannen et al. (ICLR 2020)
- [40] $\to$ Gupta & Srikumar X-FACT (ACL 2021)
- [41] $\to$ Ribeiro et al. LIME (ACM SIGKDD 2016)
- [45] $\to$ Jung et al. AASIST (Interspeech 2022)
- [47] $\to$ Dolhansky et al. DFDC (arXiv 2020)
- [49] $\to$ Tramèr et al. (ICLR 2018)
- Added: He et al. DeBERTa-v3 (2021), Schlichtkrull et al. AVeriTeC (2023), Feng et al. TwiBot-20 (2021), Chow (1970), Platt (1999).

---

## 10. Figures & Tables Inventory

| Original Item | Status | Action in Revised Manuscript | Replacement File Path |
| :---: | :---: | :--- | :--- |
| **Figure 15** | Garbled | Replaced with clean track-level ablation bar chart | `milestones/M15_release/figures/fig15_multimodal_ablation_matrix.png` |
| **Figure 16** | Garbled | Replaced with reliability curves & confidence histograms | `milestones/M15_release/figures/fig16_calibration_reliability_diagrams.png` |
| **Figure 19** | Illustrative | Replaced with empirical regional threat heatmap $T(r)$ | `milestones/M14_service_ops/figures/fig19_regional_threat_heatmap.png` |
| **Figure 22** | Fabricated | Replaced with parameterized $M/M/c$ queue simulation [SIMULATED] | `milestones/M15_release/figures/fig22_queueing_sensitivity_simulation.png` |
| **Figure 28-29** | Illustrative | Replaced with empirical CUDA token-scaling plots ($\alpha = 0.040$) | `milestones/M13_systems_perf/figures/fig28_latency_vs_tokens.png` |
| **Table 7** | Garbled | Replaced with clean LaTeX booktabs track metrics | `milestones/M15_release/tables/table07_calibrated_track_metrics.tex` |
| **Table 8** | Garbled | Replaced with clean LaTeX booktabs cross-domain shift | `milestones/M15_release/tables/table08_cross_domain_generalization.tex` |
| **Table 13** | Legacy | Replaced with synchronized CUDA timing measurements | `milestones/M13_systems_perf/tables/table13_module_latency.csv` |
| **Table 22** | Impossible | Replaced with RoBERTa position limit exception audit ($n \le 512$) | `milestones/M13_systems_perf/tables/table22_scaling_fits.csv` |
