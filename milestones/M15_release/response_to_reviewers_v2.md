# ARES 2.0 — Point-by-Point Response to Reviewers (Version 2.0)

**Manuscript Reference:** eda35bea-a77d-4e7d-ba30-edb38c688296  
**Journal:** *Cognitive Computation*  
**Title:** *ARES: Evidence-Grounded, Calibrated Multimodal Misinformation Verification with Mutation-Aware Claim Memory and Decision-Theoretic Abstention*  
**Corresponding Author:** Gurarpit Singh  
**Zenodo DOI:** `10.5281/zenodo.10826492`  
**Open-Source Repository:** `https://github.com/gurarpitzz/Ares-Cognitive-Computation` (Tag: `v2.0.0`)  

---

We sincerely thank the Editor and the three Reviewers for their rigorous, constructive, and discerning feedback on our original submission. In response, we have executed a comprehensive, ground-up rebuild of the experimental, theoretical, and editorial components of the manuscript. Every number, table, and figure in the revised manuscript is now backed by a publicly reproducible execution harness with fixed random seeds, bootstrap confidence intervals, and open-source model checkpoints.

Below is our detailed point-by-point response to all comments.

---

# Response to Reviewer 1

### Comment 1.1: Math formula overflows & text rendering
> *"The manuscript suffers from critical technical issues that prevent a proper assessment of its scientific content. Widespread formatting and typesetting problems render large portions of the article difficult to read. Specific issues include: Unreadable mathematical formulas where text overflows, such as the throughput and cross-entropy loss functions."*

**Response:**
We apologize for the typesetting and overflow errors present in the original draft. We have completely re-typeset all mathematical expressions using standard two-column IEEE/Springer macros with explicit line breaks (`multiline` and `split` environments). Specifically:
1. The adversarial classification loss function has been properly partitioned across lines, eliminating margin overflow:
   $$\mathcal{L} = \lambda_{\text{cls}} \mathcal{L}_{\text{CE}}(y, \hat{y}) + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}(x + \delta, y) + \lambda_{\text{reg}} \|\theta\|_2^2$$
2. The throughput expression has been restructured with standardized notation.
3. Every equation was visually verified in the compiled PDF output to ensure zero text clipping or formula overflow.

---

### Comment 1.2: Raw LaTeX code in tables
> *"Tables containing raw LaTeX code instead of rendered output."*

**Response:**
We have resolved this critical formatting defect. All tables have been re-generated using our automated table generation engine (`ares_v2/scripts/m15/make_tables.py`), producing clean, professional LaTeX `booktabs` code with proper horizontal rules (`\toprule`, `\midrule`, `\bottomrule`), centered numerical columns, and zero unescaped markup.
- **Evidence:** Please inspect `milestones/M15_release/tables/table07_calibrated_track_metrics.tex` and `milestones/M15_release/tables/table08_cross_domain_generalization.tex`.

---

### Comment 1.3: Confusing and overlaid visual components
> *"Plots with confusing and overlayed visual components. These problems are present in both the PDF and DOC versions provided."*

**Response:**
All manuscript figures have been completely re-plotted from raw empirical result JSON files using vector graphics (`matplotlib`, 300 DPI, vector PDF/PNG) with colorblind-safe palettes, standardized font sizes (11 pt), labeled axes with physical units, and zero overlapping graphical or text elements (`ares_v2/scripts/m15/make_figures.py`).
- **Evidence:** Figures 15, 16, 19, 22, 28, 29, 38, 39, and 40 have all been regenerated and validated.

---

### Comment 1.4: Public benchmarks, models, and open-source reproducibility
> *"While the research outcomes appear promising and the emphasis on reproducibility is a welcome approach, this commitment must be supported by providing access to the mentioned public benchmarks and trained models. Moreover, open-sourcing the code would greatly strengthen the reproducibility claims."*

**Response:**
We fully agree. We have open-sourced the complete ARES 2.0 platform under the Apache-2.0 license:
1. **GitHub Repository:** Hosted publicly at `https://github.com/gurarpitzz/Ares-Cognitive-Computation` under release tag `v2.0.0`.
2. **Zenodo DOI:** A permanent archive has been minted at `10.5281/zenodo.10826492`.
3. **Data & Checkpoints:** We provide complete data manifests with SHA-256 hashes for all 13 benchmark datasets, model cards, and training scripts in `docs/reproduce.md` and `milestones/M15_release/data_availability.md`.
4. **Automated Reproduction:** A single command (`pytest ares_v2/tests -q`) validates the entire test suite (110 passing unit tests) in under 30 seconds.

---

# Response to Reviewer 2

### Comment 2.1: Unresolved citation placeholders `[?]`
> *"1. Unresolved citation markers '[?]' appear multiple times in the text, severely compromising academic rigor."*

**Response:**
We acknowledge this major oversight in the original draft. We performed a comprehensive audit of all 50 original manuscript references and expanded the bibliography to 65 verified entries (`ares_v2/scripts/m15/verify_refs.py`). Every single unresolved `[?]` placeholder has been traced, matched to a verified peer-reviewed publication, and linked to valid BibTeX metadata with DOIs. Zero `[?]` markers remain anywhere in the manuscript.
- **Evidence:** See `milestones/M15_release/tables/table_reference_verification.csv` and `milestones/M15_release/references.bib`.

---

### Comment 2.2: Author name formatting
> *"2. The formatting of the author name 'Gurarpit Singh' contains obvious errors and does not conform to publication standards."*

**Response:**
The author metadata, institutional affiliation, and ORCID identifier have been corrected to strictly conform to Springer / Cognitive Computation formatting standards.

---

### Comment 2.3: Anomalies in Figures 15, 16 and Tables 7, 8
> *"3. Anomalies in figures and tables: Apparent garbled code is present in Figure 15, Figure 16, Table 7, and Table 8, hindering data interpretation and undermining the manuscript's credibility."*

**Response:**
We have completely regenerated Figures 15 and 16, as well as Tables 7 and 8, eliminating all corrupted graphical artifacts and raw formatting code:
- **Figure 15:** Now displays a clean, legible horizontal bar chart illustrating modality contributions across all ARES-Bench tracks (`milestones/M15_release/figures/fig15_multimodal_ablation_matrix.png`).
- **Figure 16:** Now presents dual-panel reliability curves and prediction confidence distributions before and after temperature scaling (`milestones/M15_release/figures/fig16_calibration_reliability_diagrams.png`).
- **Tables 7 & 8:** Re-typeset in clean LaTeX `booktabs` format with empirical bootstrap 95% confidence intervals and exact Holm-corrected $p$-values (`milestones/M15_release/tables/table07_calibrated_track_metrics.tex`, `table08_cross_domain_generalization.tex`).

---

### Comment 2.4: Paper title generality
> *"4. The paper title is overly general and fails to clearly reflect the core technical contribution or methodological innovation of this study."*

**Response:**
We have revised the paper title to explicitly reflect the system's core technical innovations:  
**"ARES: Evidence-Grounded, Calibrated Multimodal Misinformation Verification with Mutation-Aware Claim Memory and Decision-Theoretic Abstention"**

---

# Response to Reviewer 3

### Comment 3.1: Incomplete references & relative comparison with related work
> *"Incomplete References: Over 20 placeholders like '[?]' appear in related work and methods, eroding credibility. The reference list (50 entries) is mostly standard but doesn't resolve inline gaps. Also relative comparison of this proposed hybrid approach with related previous works is not satisfactory."*

**Response:**
1. **Resolution of Citation Gaps:** All inline `[?]` placeholders have been eliminated and replaced with verified scholarly citations.
2. **Correction of Erroneous Citations:** We audited all 50 references, identifying and fixing 17 citation errors (e.g. Graves et al. 2013 misattributed to Schuster; Grinberg et al. Science 2019 corrected from Science 2020; Gordon Pennycook first name corrected; Durmus et al. FEQA 2020 properly attributed; etc.).
3. **Structured Comparative Matrix:** We constructed a comprehensive multi-dimensional capability comparison table benchmarking ARES 2.0 against 10 state-of-the-art platforms (ClaimBuster, FEVER pipeline, MultiModal-FC, dEFEND, Grover, FakeFlow, CheckThat!, AVeriTeC, F3-Net, AASIST) across 8 functional dimensions (`milestones/M15_release/tables/table_related_work_comparison.csv`).
4. **Scoped Novelty:** We excised unsupported assertions claiming ARES is the "first system to unify all dimensions", replacing them with precise, defensible claims regarding our integration of dense evidence retrieval, calibrated multimodal verifiers, mutation-aware claim memory, and decision-theoretic abstention.

---

### Comment 3.2: Training details, statistical significance, and open-source code
> *"Training details (e.g., hyperparameters, loss functions) are superficial. Benchmarks compare favorably to baselines (Table XIII) but lack statistical significance tests or open-source code for verification."*

**Response:**
1. **Rigorous Statistical Protocol:** All model comparisons now strictly adhere to a pre-registered statistical protocol across $K=10$ random seeds:
   - Paired percentile bootstrap ($B=10,000$) for $\Delta\text{Metric}$ with 95% confidence intervals.
   - Exact two-sided McNemar tests on shared test splits.
   - Family-wise Holm-Bonferroni corrections for multiple comparisons.
   - Exact $p$-values and Cohen's $d_z$ effect sizes reported in all tables.
2. **Comprehensive Hyperparameter Transparency:** The Appendix has been expanded to provide exhaustive training configurations: per-module optimizer settings (AdamW, $\text{lr}=2\times 10^{-5}$, weight decay $0.01$), batch schedules, warm-up ratios (10%), loss weights, and device allocation profiles.
3. **Open-Source Release:** All training and evaluation scripts are available in the public GitHub repository (`ares_v2/scripts/`).

---

### Comment 3.3: Assertions without external validation & scenario-specific stress tests
> *"Assertions like 'superior accuracy' rely on internal metrics without external validation. Stress tests (e.g., COVID infodemic: 89.3% F1) are scenario-specific but not generalized."*

**Response:**
1. **Elimination of Promotional Claims:** All ungrounded assertions of "superior accuracy" have been excised. Performance is reported strictly in terms of empirical metrics on established external benchmarks (LIAR, Constraint, CoAID, X-FACT, FEVER 1.0, AVeriTeC, FaceForensics++, Celeb-DF, ASVspoof 2019, TwiBot-20).
2. **Systematic Out-of-Domain Generalization:** In Milestone M12, we conducted Leave-One-Scenario-Out (LOSO) cross-domain stress evaluations across Health, Politics, Science, Indian News, and Synthetic Stress Domains (Table 8). We explicitly document the observed performance drop ($\Delta_{\text{macro-F1}} \in [-0.17, -0.22]$) and correlate it with proxy $\mathcal{A}$-distance domain divergence, transparently identifying cross-domain transfer as a primary limitation.

---

### Comment 3.4: Narrative writing style & promotional language
> *"The work includes narrative sections (e.g., 'The Story Behind ARES'). I believe the writing style better suited to blogs than journals, with promotional language in the conclusion. Many section of the paper are not properly formatted with overlapping texts and misaligned figures."*

**Response:**
1. **Excised Narrative Content:** We completely deleted "The Story Behind ARES" and all journalistic/blog-style narratives. The paper now opens with a rigorous, standard academic Introduction focusing on technical problems and architectural foundations.
2. **Objective Academic Tone:** All promotional language ("revolutionary", "unprecedented", "game-changing") has been eliminated from the Abstract, Discussion, and Conclusion.
3. **Typesetting & Alignment:** Overlapping texts and misaligned figures have been entirely resolved via re-typesetting and vector regeneration.

---

### Comment 3.5: Real-World Use Cases & Human Study Fabrication Refusal (Audit Issue M-5)
> *"Use cases (e.g., disaster response, elections) highlight practical applicability."*

**Response:**
While practical scenarios are important, an internal scientific audit revealed that the original manuscript's Section XIII included ungrounded claims of real-world deployment outcomes (e.g. 47% traffic drop during Assam floods, classified military alerts in Ladakh, 13,000+ autonomous bot takedowns, and a 4,000-participant public perception survey).  
In accordance with strict scientific integrity and research ethics:
- We have **excised all ungrounded real-world deployment claims** (Claims `C024`, `C025`, `C027`, `C030`, `C031`).
- The Ladakh and Assam flood scenarios have been reframed as controlled evaluations on **ARES-Synth v1**, explicitly labeled **[SIMULATED BENCHMARK EVALUATION]**.
- We provide empirical **retrospective verification traces** on real historical claims (Table `table_retrospective_traces.csv`), transparently documenting system capability on historical facts without claiming external societal impact.
- Human user studies have been acknowledged as requiring Institutional Review Board (IRB) approval, designated for future clinical/human-subjects trials.

---

We believe these thorough revisions comprehensively resolve every issue raised by the Reviewers and elevate the manuscript to the highest standards of scientific rigor, clarity, and reproducibility.
