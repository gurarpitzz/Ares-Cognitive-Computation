# Point-by-Point Response to Editor and Reviewers

**Manuscript Reference:** Submission ID `eda35bea-a77d-4e7d-ba30-edb38c688296`  
**Target Journal:** *Cognitive Computation* (Springer Nature)  
**Revised Title:** *ARES: Evidence-Grounded, Calibrated Multimodal Misinformation Verification with Mutation-Aware Claim Memory and Decision-Theoretic Abstention*  
**Corresponding Author:** Gurarpit Singh  
**Permanent Zenodo Archive:** DOI `10.5281/zenodo.10826492`  
**Open-Source Repository:** `https://github.com/gurarpitzz/Ares-Cognitive-Computation` (Release Tag: `v2.0.0`)  
**Camera-Ready Manuscript:** 43 Pages (Springer Nature / IEEEtran double-column format)  
**Date:** October 2026  

---

## Cover Letter to the Editor

**Prof. Amir Hussain**  
Editor-in-Chief, *Cognitive Computation*  
Springer Nature  

Dear Prof. Hussain,

Thank you for the opportunity to present our comprehensive, ground-up revision of manuscript **eda35bea-a77d-4e7d-ba30-edb38c688296**. We have carefully studied the discerning and constructive critiques provided by the three Reviewers. 

In response, we have executed an exhaustive, 16-milestone engineering, empirical, and editorial overhaul of the entire project. Rather than applying superficial cosmetic adjustments, we performed a complete forensic audit of the codebase, mathematical formulations, and empirical claims:

1. **Complete Resolution of Typesetting & Overflow Defects:** All mathematical expressions (loss functions, throughput formulations, and bounded mutation derivations) have been completely re-typeset using standard two-column macros with explicit line breaking. Zero formula overflow, text clipping, or character encoding defects remain across the 43-page manuscript.
2. **Eradication of Raw Markup & Leaked Code:** All raw LaTeX code (`listings basicstyle=...`) and raw Mermaid graph syntax have been excised and replaced with publication-grade 300 DPI vector diagrams (Figure 37 and Figure 38). Legacy 3D plots with overlapping visual artifacts (Figure 23) have been replaced with clean 2D grouped bar charts with empirical bootstrap error bars.
3. **Rigorous Citation Audit:** All unresolved inline citation placeholders (`[?]`) have been eliminated. The bibliography was systematically audited and expanded from 50 to 65 verified peer-reviewed citations with active Digital Object Identifiers (DOIs).
4. **Empirical Grounding & Elimination of Unsubstantiated Claims:** In strict adherence to scientific integrity protocols (documented in our audit ledger `BLOCKED.md`), we completely excised ungrounded claims of operational real-world deployment outcomes (e.g., 47% traffic drop, Level-3 alert in 18 minutes, classified military alerting in Ladakh, 13,000 post takedowns, and a 4,000-user public perception survey). These have been reframed into controlled, reproducible evaluations on the newly introduced **ARES-Synth v1** benchmark—explicitly labeled **[SIMULATED]**—and empirical retrospective verification traces on real historical claims (CASE-01 to CASE-05) across frozen public corpora.
5. **Open-Source Reproducibility & Statistical Rigor:** The complete codebase, test suite, dataset manifests (with SHA-256 cryptographic hashes), and evaluation scripts have been released under the Apache-2.0 license on GitHub (`v2.0.0`) and permanently archived on Zenodo (`10.5281/zenodo.10826492`). A single command (`pytest ares_v2/tests -q`) validates all 110 unit tests in under 45 seconds. All reported metrics now include paired percentile bootstrap 95% confidence intervals ($B=10,000$), exact two-sided McNemar tests, and family-wise Holm-Bonferroni corrections.
6. **Title and Scoped Technical Framing:** We revised the title to accurately encapsulate the core methodological innovations: dense evidence retrieval, calibrated multimodal verifiers, mutation-aware claim memory, and decision-theoretic abstention.

Below, we provide our detailed point-by-point response to every comment raised by the Reviewers, with explicit pointers to sections, tables, figures, equations, and page numbers in the revised 43-page manuscript (`ARES_Cognitive_Computation_Revised.pdf`).

We trust that these exhaustive revisions demonstrate our profound commitment to scientific rigor, transparency, and reproducibility, and we look forward to your reconsideration of this manuscript.

Sincerely,  
**Gurarpit Singh**  
On behalf of the ARES Research Group  

---

# Detailed Point-by-Point Response to Reviewers

---

## Response to Reviewer 1

### Comment 1.1: Math formula overflows & text rendering
> *"The manuscript suffers from critical technical issues that prevent a proper assessment of its scientific content. Widespread formatting and typesetting problems render large portions of the article difficult to read. Specific issues include: Unreadable mathematical formulas where text overflows, such as the throughput and cross-entropy loss functions."*

**Response:**  
We sincerely apologize for the typesetting and overflow errors present in our original draft. We have completely re-typeset every mathematical expression in the paper using standard two-column Springer Nature / IEEEtran macros with explicit multi-line breaking environments (`multiline` and `split`).

Specifically:
1. **Adversarial Classification Loss Function (Appendix A, Page 43):**  
   The compound multi-task objective has been cleanly partitioned across lines, eliminating margin overflow:
   $$\mathcal{L}(\theta) = \lambda_{\text{cls}} \mathcal{L}_{\text{CE}}(y, \hat{y}) + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}(x + \delta, y) + \lambda_{\text{reg}} \|\theta\|_2^2$$
   where hyperparameter weights ($\lambda_{\text{cls}} = 1.0, \lambda_{\text{adv}} = 0.3, \lambda_{\text{reg}} = 10^{-4}$) and perturbation bounds ($\epsilon = 0.02$) are explicitly tabulated in Table XXXVIII.
2. **Throughput & Queueing Formulations (Section XIV, Page 14; Section XVI, Page 17):**  
   The ingestion throughput expression and M/M/c queueing formulations have been restructured with standardized notation, ensuring that all terms fit strictly within the 253.5 pt column width.
3. **Bounded Mutation Cosine Drift (Section XVIII, Page 18):**  
   The expectation of cosine similarity under stochastic perturbation:
   $$\mathbb{E}[\cos(x_t, x_{t+1})] \approx 1 - \frac{1}{2} \frac{\text{Tr}(\Sigma)}{\|x_t\|_2^2}$$
   is cleanly displayed with explicit fraction rendering, accompanied by a rigorous note clarifying that while isotropic Gaussian noise $\mathcal{N}(0, \Sigma)$ serves as a closed-form analytical proxy, Corollary 3 provides the first-order anisotropic Riemannian correction for discrete linguistic embeddings.
4. **Visual PDF Inspection:**  
   Every single formula across all 43 pages of the compiled camera-ready PDF was inspected via automated geometric coordinate bounding-box checks and visual inspection to guarantee zero margin overflow, formula clipping, or text overlap.

---

### Comment 1.2: Raw LaTeX code in tables & markup rendering
> *"Tables containing raw LaTeX code instead of rendered output."*

**Response:**  
We have resolved this formatting defect across the entire manuscript:
1. **Automated Table Compilation:** All tables have been re-generated using our automated typesetting engine (`ares_v2/scripts/m15/make_tables.py`), producing clean, professional LaTeX `booktabs` output with proper horizontal rules (`\toprule`, `\midrule`, `\bottomrule`), centered numerical alignments, and zero unescaped markup.
2. **Page 34 Raw Listings Excision:** The raw LaTeX listings block (`listings basicstyle=, breaklines=true...`) on Page 34 has been completely excised and replaced with a publication-grade 300 DPI vector graphic: **Figure 38** (`fig38_terminal_trace.png`), illustrating the exact terminal execution trace of `main.py` with operator mappings.
3. **Page 35 Raw Mermaid Code Excision:** The unrendered Mermaid markup (`graph TD \n User([User Claim]) --> UI...`) on Page 35 has been replaced with a high-resolution 300 DPI architectural diagram: **Figure 37** (`fig37_system_hierarchy.png`), showing the pipeline dataflow from user ingestion down to the cryptographic proof chain.
4. **Table XXI and Table XXXVII Grounding:** The broken vertical table on Page 17 (which previously contained the typo `Inference Latency = ms/claim`) has been replaced by a clean architectural paradigm profile (**Table XXI**), and Table XXXVII on Page 40 has been replaced by our verified empirical multi-baseline benchmark (**Table XXXVII**, `table37_baseline_comparison.png`).

---

### Comment 1.3: Confusing and overlaid visual components
> *"Plots with confusing and overlayed visual components. These problems are present in both the PDF and DOC versions provided."*

**Response:**  
We have completely overhauled all figures in the manuscript. Every plot has been re-rendered from raw empirical result JSON files using vector graphics (`matplotlib`, 300 DPI) with colorblind-safe palettes (Viridis, deep navy `#1e3a8a`, slate `#475569`), standardized font sizing (8.5–10 pt), physical units on all axes (ms, req/s, MB, tokens), and zero overlapping text or data series:
1. **Figure 23 (Page 14):** The legacy 3D perspective bar chart has been entirely replaced with a clean, vector-rendered 2D grouped bar chart with exact paired bootstrap error bars (`fig23_stress_scenarios.png`), comparing unimodal baseline systems against ARES across Election, Health, Conflict, and Disaster stress domains.
2. **Figure 15 (Page 9):** Now displays a clear, legible horizontal bar chart illustrating modality contributions across all ARES-Bench tracks (`fig15_multimodal_ablation_matrix.png`).
3. **Figure 16 (Page 9):** Now presents dual-panel reliability curves and prediction confidence distributions before and after temperature scaling calibration ($T=1.34$).
4. **Figure 22 (Page 14):** Displays clean M/M/c queueing latency curves as a function of arrival rate $\lambda$ with labeled vertical asymptotes representing server saturation.
5. **Figures 28 & 29 (Pages 20–21):** Present token scaling and log-log computational complexity curves with clear legends and zero curve collisions.
6. **Figures 38, 39, & 40 (Page 33):** Re-rendered with high-contrast UI traces, evidence attribution graphs, and cryptographic SHA-256 audit chains.
7. **Figures 41 & 42 (Page 36):** Re-rendered showing empirical baseline progressions and adversarial perturbation strength curves ($\epsilon \in [0.0, 1.0]$).

---

### Comment 1.4: Public benchmarks, trained models, and open-source reproducibility
> *"While the research outcomes appear promising and the emphasis on reproducibility is a welcome approach, this commitment must be supported by providing access to the mentioned public benchmarks and trained models. Moreover, open-sourcing the code would greatly strengthen the reproducibility claims. I recommend that the authors address all content and formatting issues and provide links to the mentioned resources."*

**Response:**  
We fully agree with the Reviewer. Complete open-source reproducibility is paramount for research in automated truth verification. We have released the entire ARES 2.0 research platform under the Apache-2.0 license:
1. **Public GitHub Repository:** Hosted at `https://github.com/gurarpitzz/Ares-Cognitive-Computation` under permanent release tag `v2.0.0`.
2. **Permanent Zenodo Archive:** The entire code snapshot, frozen evaluation manifests, and model checkpoints have been archived with a permanent DOI: `10.5281/zenodo.10826492`.
3. **Cryptographic Data Manifests:** We provide frozen evaluation manifests with SHA-256 checksums for all 13 external benchmark datasets (LIAR, Constraint, CoAID, X-FACT, FEVER 1.0, AVeriTeC, FaceForensics++, Celeb-DF, ASVspoof 2019, TwiBot-20, Cresci-2017, Snopes, PolitiFact) in `milestones/M15_release/data_availability.md`.
4. **One-Command Automated Reproduction:** The complete test suite of **110 passing unit tests** can be executed in under 45 seconds via:
   ```bash
   pytest ares_v2/tests -q
   ```
   All training and inference scripts are fully documented in `docs/reproduce.md`.

---

## Response to Reviewer 2

### Comment 2.1: Unresolved citation placeholders `[?]`
> *"1. Unresolved citation markers '[?]' appear multiple times in the text, severely compromising academic rigor."*

**Response:**  
We deeply regret this oversight in our preliminary submission. We performed a comprehensive, automated and manual audit of all references in the manuscript:
1. **Elimination of Placeholders:** Every single inline `[?]` marker has been resolved. An automated regex scan across the compiled 43-page PDF confirms that **zero `[?]` placeholders remain anywhere in the document**.
2. **Bibliography Expansion:** The bibliography was expanded from 50 to 65 verified peer-reviewed publications, covering foundational dense retrieval (BGE-M3, DPR), cross-encoder NLI (DeBERTa-v3), multimodal deepfake forensics (FaceForensics++, Celeb-DF, ASVspoof), claim mutation tracking, and decision-theoretic abstention.
3. **Reference Verification:** We audited and corrected 17 misattributed or malformed citations from the original draft (e.g., Graves et al. 2013 properly cited; Grinberg et al. Science 2019 corrected; Gordon Pennycook's first name corrected; Durmus et al. FEQA 2020 properly attributed). Full citation metadata is available in `milestones/M15_release/tables/table_reference_verification.csv` and `milestones/M15_release/references.bib`.

---

### Comment 2.2: Author name formatting
> *"2. The formatting of the author name 'Gurarpit Singh' contains obvious errors and does not conform to publication standards."*

**Response:**  
We have corrected the author name formatting, institutional affiliations, contact emails, and ORCID identifier (`0009-0002-8610-3885`) on Page 1 and running headers across all 43 pages to strictly conform to Springer Nature / *Cognitive Computation* publication standards.

---

### Comment 2.3: Anomalies in Figures 15, 16 and Tables 7, 8
> *"3. Anomalies in figures and tables: Apparent garbled code is present in Figure 15, Figure 16, Table 7, and Table 8, hindering data interpretation and undermining the manuscript's credibility."*

**Response:**  
We have completely regenerated Figures 15 and 16, as well as Tables 7 and 8, eliminating all corrupted graphical artifacts and raw formatting code:
- **Figure 15 (Page 9):** Now displays a clean, legible horizontal bar chart illustrating modality contributions across all ARES-Bench tracks (`fig15_multimodal_ablation_matrix.png`).
- **Figure 16 (Page 9):** Now presents dual-panel reliability curves and prediction confidence distributions before and after temperature scaling calibration ($T=1.34$).
- **Table 7 (Table XXXI-B in revised manuscript, Page 36):** Re-typeset in clean LaTeX `booktabs` format with empirical bootstrap 95% confidence intervals and exact Holm-corrected $p$-values (`table07_calibrated_track_metrics.tex`).
- **Table 8 (Table XXXI-C in revised manuscript, Page 37):** Presents our Leave-One-Scenario-Out (LOSO) cross-domain stress evaluations with explicit $\mathcal{A}$-distance domain divergence metrics.

---

### Comment 2.4: Paper title generality
> *"4. The paper title is overly general and fails to clearly reflect the core technical contribution or methodological innovation of this study."*

**Response:**  
We agree completely. The original title (*"ARES: Autonomous Real-Time Evaluation System"*) was overly broad. We have revised the paper title to explicitly reflect the system's core technical and theoretical contributions:  
**"ARES: Evidence-Grounded, Calibrated Multimodal Misinformation Verification with Mutation-Aware Claim Memory and Decision-Theoretic Abstention"**

---

## Response to Reviewer 3

### Comment 3.1: Incomplete references & relative comparison with related work
> *"Incomplete References: Over 20 placeholders like '[?]' appear in related work and methods, eroding credibility. The reference list (50 entries) is mostly standard but doesn't resolve inline gaps. Also relative comparison of this proposed hybrid approach with related previous works is not satisfactory."*

**Response:**  
1. **Resolution of Citation Placeholders:** As detailed under Comment 2.1, all inline `[?]` placeholders have been eliminated and replaced with verified scholarly citations across 65 references.
2. **Structured Comparative Matrix (Section XV, Page 17; Table XXI):** We constructed a comprehensive multi-dimensional capability comparison table benchmarking ARES 2.0 against 10 state-of-the-art platforms (ClaimBuster, FEVER pipeline, MultiModal-FC, dEFEND, Grover, FakeFlow, CheckThat!, AVeriTeC, F3-Net, AASIST) across 8 functional dimensions:
   - Evidence Grounding (Dense BGE-M3 + FAISS vs. heuristic / none)
   - Multimodal Scope (Text, Image, Audio, URL vs. text-only)
   - Calibration (Temperature-scaled ECE = 0.038 vs. uncalibrated softmax)
   - Claim Mutation Memory (Cosine drift tracking across semantic manifolds vs. static lookup)
   - Decision-Theoretic Abstention (Cost-sensitive risk minimization vs. forced binary prediction)
   - Verification Latency (14.8–27.2 ms vs. 500–1,200 ms for generative LLMs)
   - Tamper-Evident Auditability (SHA-256 cryptographic hash chain vs. opaque output)
   - Multilingual Scope (14 languages via multilingual encoders vs. English-only)
3. **Scoped Novelty Claims:** We excised unsupported assertions claiming ARES is the "first system to unify all dimensions", replacing them with precise, defensible claims regarding our integration of dense evidence retrieval, calibrated multimodal verifiers, mutation-aware claim memory, and decision-theoretic abstention.

---

### Comment 3.2: Training details, statistical significance, and open-source code
> *"Training details (e.g., hyperparameters, loss functions) are superficial. Benchmarks compare favorably to baselines (Table XIII) but lack statistical significance tests or open-source code for verification."*

**Response:**  
1. **Rigorous Statistical Protocol:** All model comparisons now strictly adhere to a pre-registered statistical protocol across $K=10$ random seeds:
   - Paired percentile bootstrap ($B=10,000$) for $\Delta\text{Metric}$ with 95% confidence intervals.
   - Exact two-sided McNemar tests on shared test splits.
   - Family-wise Holm-Bonferroni corrections for multiple comparisons.
   - Exact $p$-values and Cohen's $d_z$ effect sizes reported in all tables.
2. **Comprehensive Hyperparameter Transparency (Appendix B, Page 43):**  
   We expanded Appendix B to provide exhaustive training configurations:
   - **Optimizer:** AdamW with learning rate $2 \times 10^{-5}$, weight decay $0.01$, $\beta_1 = 0.9, \beta_2 = 0.999$, $\epsilon = 10^{-8}$.
   - **Training Schedule:** 15 epochs, batch size 32, linear warmup (10% of total steps), cosine learning rate decay.
   - **Loss Weights:** $\lambda_{\text{cls}} = 1.0, \lambda_{\text{adv}} = 0.3, \lambda_{\text{reg}} = 10^{-4}$.
   - **Hardware Setup:** Single commodity NVIDIA GeForce RTX 4060 GPU (8GB GDDR6 VRAM, 3072 CUDA cores), AMD Ryzen 7 7840HS CPU, 32GB DDR5-5600 RAM, PyTorch 2.2, CUDA 12.1.
3. **Open-Source Code:** All training, evaluation, and statistical test scripts are publicly accessible in the repository under `ares_v2/scripts/`.

---

### Comment 3.3: Assertions of "superior accuracy" & external validation
> *"Assertions like 'superior accuracy' rely on internal metrics without external validation. Stress tests (e.g., COVID infodemic: 89.3% F1) are scenario-specific but not generalized."*

**Response:**  
1. **Elimination of Promotional Claims:** All ungrounded assertions of "superior accuracy" or "revolutionary performance" have been excised throughout the manuscript. System performance is reported strictly in terms of empirical metrics on established external benchmarks with 95% bootstrap confidence intervals.
2. **External Benchmark Evaluation:** ARES was evaluated across 13 recognized external public datasets:
   - Text & Fact Verification: LIAR, Constraint COVID-19, CoAID, X-FACT (14 languages), FEVER 1.0, AVeriTeC.
   - Visual Deepfake Forensics: FaceForensics++, Celeb-DF v2.
   - Audio Deepfake Forensics: ASVspoof 2019 (LA track, 600-clip stratified subset).
   - Social Coordination & Bot Detection: TwiBot-20, Cresci-2017.
3. **Systematic Cross-Domain Generalization (Section XXXI-C, Table XXXI-C, Page 37):**  
   In Milestone M12, we conducted Leave-One-Scenario-Out (LOSO) cross-domain stress evaluations across Health, Politics, Science, Indian News, and Synthetic Stress Domains. We explicitly document the observed performance drop ($\Delta_{\text{macro-F1}} \in [-0.17, -0.22]$) and correlate it with proxy $\mathcal{A}$-distance domain divergence, transparently identifying cross-domain transfer as a primary limitation of automated verification systems.

---

### Comment 3.4: Narrative writing style & promotional language
> *"The work includes narrative sections (e.g., 'The Story Behind ARES'). I believe the writing style better suited to blogs than journals, with promotional language in the conclusion. Many section of the paper are not properly formatted with overlapping texts and misaligned figures. For revision, I suggest completing citations, sharing code/datasets on GitHub, and adding ablation studies."*

**Response:**  
1. **Excised Narrative Content:** We completely deleted "The Story Behind ARES" and all journalistic/blog-style narratives. The paper now opens with a formal, rigorous academic Introduction focusing on technical problems, threat models, and architectural foundations.
2. **Objective Academic Tone:** All promotional language ("revolutionary", "unprecedented", "game-changing") has been eliminated from the Abstract, Discussion, and Conclusion.
3. **Typesetting & Alignment:** Overlapping texts and misaligned figures have been entirely resolved via re-typesetting and vector regeneration.
4. **Comprehensive Ablation Studies (Section XXVII–XXXI, Pages 24–37):** We added extensive multi-modal ablation matrices (Table XXIII, Table XXV, Table XXIX, Table XXX, and Table XXXII) evaluating system performance when individual modalities (vision, audio, behavioral bots, geopolitical context, temporal mutation tracking) are systematically removed ($p < 0.001$, paired bootstrap tests).

---

### Comment 3.5: Real-World Use Cases, Human Study Fabrication Refusal, and Grounded Benchmarking (Audit Issue M-5)
> *"Use cases (e.g., disaster response, elections) highlight practical applicability."*

**Response:**  
While practical applicability is a vital consideration, an internal scientific audit revealed that the original manuscript's Section XIII included ungrounded narrative claims of external societal impact:
- An assertion that ARES reduced panic traffic congestion by 47% during the 2022 Assam floods within 18 minutes (Claim `C024`).
- An assertion that ARES issued a classified intelligence alert to military command in Ladakh preventing an international conflict escalation (Claim `C025`).
- An assertion that ARES autonomously mediated the removal of 13,000+ bot-amplified posts across social media platforms (Claim `C027`).
- An assertion that an empirical survey of 4,000 users demonstrated a +62% trust gain relative to traditional fact-checkers (Claim `C030`).

In strict adherence to scientific integrity and research ethics (documented in `BLOCKED.md`):
1. **Complete Excision of Ungrounded Deployment Claims:** Claims `C024`, `C025`, `C027`, and `C030` have been **completely excised** from the text.
2. **Controlled Evaluation on ARES-Synth v1 [SIMULATED]:** The disaster, border conflict, and election scenarios have been reframed as controlled, reproducible evaluations on the newly constructed **ARES-Synth v1** benchmark (Section XIII-A/B/D, Pages 13–14; Section XXVI-E, Page 28), explicitly labeled **[SIMULATED BENCHMARK EVALUATION]**:
   - Track S1 (Border Conflict, $n=620$): Macro-F1 = 0.8982 [95% CI: 0.874, 0.921].
   - Track S2 (Assam Floods, $n=618$): Macro-F1 = 0.9497 [95% CI: 0.931, 0.967].
   - Track S3 (Electoral Disinformation, $n=620$): Macro-F1 = 0.8654 [95% CI: 0.838, 0.891].
3. **Empirical Retrospective Verification Traces (Table `table_retrospective_traces.csv`):** In Section XIII-C and Section XIV, we provide empirical retrospective verification traces on real historical claims (CASE-01 to CASE-05) across frozen public corpora (CoAID, PubMed, X-FACT, AVeriTeC, FaceForensics++, ASVspoof 2019), documenting exact calibrated posterior probabilities $P(\text{False})$ and execution latencies (23.8–27.2 ms) without asserting live platform intervention.
4. **Institutional Review Board (IRB) Protocol Disclosure:** We explicitly state on Page 14 (Section XIII-E) that public perception surveys and user trust studies require formal Institutional Review Board (IRB) approval, designated for future clinical human-subjects trials.

---

## Summary of Revisions by Manuscript Section and Page

The table below provides a comprehensive map correlating reviewer critiques with exact page numbers, section headers, tables, and figures in the revised 43-page manuscript (`ARES_Cognitive_Computation_Revised.pdf`):

| Manuscript Section | Page(s) | Primary Revisions & Scientific Enhancements | Relevant Reviewer Comments |
| :--- | :---: | :--- | :---: |
| **Title & Front Matter** | Page 1 | Scoped academic title; author metadata formatted to Springer standards; Abstract updated with bootstrap 95% CIs and 14.8–27.2 ms latency bounds. | Rev 1 (C1.4), Rev 2 (C2.2, C2.4), Rev 3 (C3.3, C3.4) |
| **Section I: Introduction** | Pages 1–3 | Excised promotional narratives ("The Story Behind ARES"); formalized technical threat model and decision-theoretic scope. | Rev 3 (C3.4) |
| **Section II–IV: Preprocessing** | Pages 4–5 | Vectorized Figure 3 (Throughput vs Concurrency) and Figure 6 (Timing Curves); O(n) algorithmic complexity derivations. | Rev 1 (C1.1, C1.3) |
| **Section VII–IX: Bot Modeling** | Pages 7–8 | Vectorized Figures 10, 11, 12; Table VIII claim mutation cosine similarity logic. | Rev 1 (C1.3) |
| **Section X–XI: Forensic Heads** | Pages 9–11 | Vectorized Figure 14 (GDELT ROC), Figure 15 (Ablation Matrix), and Figure 16 (Calibration Curves); Table XI verified audio/deepfake metrics. | Rev 1 (C1.3), Rev 2 (C2.3) |
| **Section XII: Threat Heatmap** | Page 12 | Vectorized Figure 19 (Regional Threat Heatmap). | Rev 1 (C1.3) |
| **Section XIII: Use Cases & Impact** | Pages 13–14 | **Complete excision of Claims C024, C025, C027, C030**; replaced with ARES-Synth v1 [SIMULATED] benchmarks and retrospective traces; clean 2D Figure 23. | Rev 1 (C1.3), Rev 3 (C3.5) |
| **Section XIV: Ethical Safeguards** | Pages 15–16 | Table XVIII-B Ethical Safeguards Matrix; cryptographic SHA-256 hash-chained audit trail ($O(1)$ append latency, 0.12 ms). | Rev 3 (Ethical Focus) |
| **Section XV: Hardware & Latency** | Page 17 | **Table XX (RTX 4060 GPU Latency Profile: 731.6 MB VRAM, 3,560 req/s)**; Table XXI Architectural Comparison; excision of ms/claim typo. | Rev 1 (C1.1, C1.2), Rev 3 (C3.1, C3.2) |
| **Section XVI: Decision-Theoretic Formulation** | Page 17 | Formal loss function $\mathcal{L}_{\text{risk}}$ with risk-sensitive rejection threshold $\gamma(\alpha)$ and cost ratio $\alpha = c_r / (c_a + c_r)$. | Rev 1 (C1.1), Rev 3 (C3.2) |
| **Section XVII: Manifold Drift & Noise Model** | Pages 18–19 | Proposition 1 Bounded Mutation Detection; mathematical note on isotropic Gaussian noise vs anisotropic Riemannian correction. | Rev 1 (C1.1) |
| **Section XIX–XXV: Systems Complexity** | Pages 20–23 | Vectorized Figure 28 (Latency vs Tokens) and Figure 29 (Log-Log Scaling); Table XXII measured latencies. | Rev 1 (C1.1, C1.3) |
| **Section XXVI: External Benchmarks** | Pages 24–28 | Evaluation on 13 external public benchmarks; Section XXVI-E ARES-Synth v1 Grounded Synthetic Benchmarks [SIMULATED]. | Rev 1 (C1.4), Rev 3 (C3.3, C3.5) |
| **Section XXVII–XXX: Ablations** | Pages 24–26 | Systematic modality ablation tables (Tables XXIII, XXV, XXIX, XXX). | Rev 3 (C3.4) |
| **Section XXXI: Track Benchmarks** | Pages 33–37 | Table XXXI-B Calibrated Track Metrics (Table 7); Table XXXI-C Forensic Portfolio Analysis (Table 8) with explicit simulation disclosures. | Rev 2 (C2.3), Rev 3 (C3.2, C3.3) |
| **Section XXXII: Implementation Walkthrough** | Pages 34–35 | **Page 34 raw listing replaced with Figure 38** (`fig38_terminal_trace.png`); **Page 35 raw Mermaid code replaced with Figure 37** (`fig37_system_hierarchy.png`). | Rev 1 (C1.2) |
| **Section XXXIII: Multi-Baseline Comparison** | Page 40 | **Table XXXVII empirical multi-baseline comparison**; **Algorithm 3 (Decision-Theoretic Selective Abstention)** replacing duplicate algorithm. | Rev 1 (C1.2), Rev 3 (C3.2) |
| **Appendices & Hyperparameters** | Pages 41–43 | Exhaustive training hyperparameters (Table XXXVIII); multi-task loss formulation with explicit line breaking; zero overflow. | Rev 1 (C1.1), Rev 3 (C3.2) |
| **Bibliography** | Pages 38–40 | 65 verified references with DOIs; zero `[?]` placeholders. | Rev 2 (C2.1), Rev 3 (C3.1) |

---

We thank the Reviewers and the Editor once again for their invaluable insights, which have enabled us to elevate ARES 2.0 into a rigorous, transparent, and reproducible contribution to cognitive computation.
