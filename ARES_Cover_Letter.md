# Formal Cover Letter for Manuscript Resubmission

**Date:** October 24, 2026  

**To:**  
**Prof. Amir Hussain**  
Editor-in-Chief, *Cognitive Computation*  
Springer Nature  

**Cc:**  
**Mr. Vincent Salvo**  
Journals Editorial Office (JEO) Assistant, Group Publishing Operations  
Springer Nature (`vincent.salvo@springernature.com`)  

**From:**  
**Gurarpit Singh**  
Department of Computer Applications  
Vivekananda Institute of Professional Studies – Technical Campus (VIPS-TC)  
Guru Gobind Singh Indraprastha University (GGSIPU), Delhi, India  
Institutional Email: `bca_04817702024_gurarpit@vipstc.edu.in`  
ORCID: `0009-0002-8610-3885`  

---

**Subject:** Submission of New Manuscript Following Editorial Guidance on Previous Manuscript ID: `eda35bea-a77d-4e7d-ba30-edb38c688296`

**Manuscript Title:** *ARES: Autonomous Real-Time Evaluation System — An AI-Driven Multi-Modal Truth Infrastructure for Misinformation Defense*  
*(Revised Subtitle / Technical Focus: Evidence-Grounded, Calibrated Multimodal Misinformation Verification with Mutation-Aware Claim Memory and Decision-Theoretic Abstention)*  

---

Dear Prof. Hussain and Mr. Salvo,

I am writing to formally submit our research manuscript, titled **"ARES: Autonomous Real-Time Evaluation System — An AI-Driven Multi-Modal Truth Infrastructure for Misinformation Defense"**, for consideration as a New Submission in *Cognitive Computation*.

### Context & Editorial Trajectory
This submission follows formal correspondence with Mr. Vincent Salvo of the Springer Nature Journals Editorial Office regarding our previous submission, **Manuscript ID: eda35bea-a77d-4e7d-ba30-edb38c688296**. In his email directive dated November 17, 2025, Mr. Salvo advised:

> *"As your previous manuscript has been rejected, you may submit your updated manuscript as new submission in the system."*

We fully recognized that the previous decision was driven by substantial typesetting, LaTeX formatting, and rendering defects in the submitted files—specifically, overflowing mathematical formulas, unrendered citation placeholders (`[?]`), raw markup leaks, misaligned plots, and the absence of publicly accessible code repositories at the time of initial evaluation. Crucially, the Reviewers also emphasized the inherent novelty, architectural vision, and potential societal value of the ARES multi-modal verification framework.

In accordance with Mr. Salvo's guidance, we have prepared this resubmission as a **New Submission**, accompanied by an exhaustive, 8-page **Point-by-Point Response to Editor and Reviewers** document (`ARES_response_to_reviewers.pdf`). Rather than offering superficial cosmetic edits, we undertook a ground-up technical, empirical, and mathematical overhaul of the entire research platform across 16 core modules.

---

### Summary of Major Enhancements & New Computations
The revised manuscript—spanning exactly 43 camera-ready pages in standard double-column publication format—incorporates the following scientific advancements:

1. **Complete Mathematical & Typesetting Rebuild:**
   Every mathematical formulation (adversarial multi-task loss, ingestion queueing models, and bounded manifold mutation derivations) has been completely re-typeset using standard two-column macros with explicit line breaking. All raw LaTeX listings (previously on Page 34) and unrendered Mermaid markup (previously on Page 35) have been excised and replaced with high-resolution 300 DPI vector graphics (**Figure 38** terminal trace and **Figure 37** system hierarchy). An automated scan confirms zero margin overflow, formula clipping, or text collisions across all 43 pages.

2. **Ethical Course-Correction & Grounded Benchmarking:**
   In strict alignment with research integrity standards and scientific ethics, we completely eliminated narrative claims of unverified operational deployment outcomes (e.g., estimated traffic congestion cuts during floods, military alerts during border standoffs, live platform post takedowns, and public perception surveys). All scenario evaluations have been systematically reframed as controlled, reproducible benchmarks on our newly constructed **ARES-Synth v1** suite ($n=1,858$ items), explicitly labeled **[SIMULATED BENCHMARK EVALUATION]**, and empirical retrospective verification traces on frozen historical fact-checking corpora (CASE-01 to CASE-05). Furthermore, Section XIII-E explicitly discloses that human trust studies require formal Institutional Review Board (IRB) approval reserved for designated clinical human-subjects trials.

3. **Complete De-Branding & Standalone Scientific Integrity:**
   All internal developmental milestones, phase labels, and auditing markers have been excised. The running headers and footers have been cleaned to present a pristine, standalone scientific monograph of the highest computational standard.

4. **Extensive New Empirical Computations & Hardware Telemetry:**
   * **Commodity Hardware Profiling:** Evaluated on a dedicated **NVIDIA GeForce RTX 4060 Laptop GPU** (8GB GDDR6 VRAM, 3072 CUDA cores), **AMD Ryzen 7 7840HS CPU**, and **32GB DDR5 RAM**. Verified single-claim sequential latency is bounded between **14.8–27.2 ms/claim**, with asynchronous batched worker throughput reaching **3,560 requests/second** at a minimal **2,352 MB** VRAM footprint (**Table XX**, Page 17).
   * **Temperature Scaling Calibration:** Implemented post-hoc Platt / Temperature scaling ($T=1.34$), reducing Expected Calibration Error from $0.1420$ down to **$\text{ECE} = 0.0312$** across 13 public test sets (**Figure 16**, Page 9; **Table 07**, Page 36).
   * **Vision & Audio Deepfake Portfolios:** Cross-dataset transfer from FaceForensics++ to Celeb-DF v2 ($2,030$ test videos, $98.4\%$ fake prevalence) demonstrated video-level **$\text{ROC-AUC} = 0.7184$ [$95\%$ CI: $0.702, 0.734$]** (**Figure 11-B**). Speech spoofing on ASVspoof 2019 LA with silence shortcut controls achieved **$\text{EER} = 2.41\%$** and **$\text{Macro-F1} = 0.9760$** (**Section XXXI-C**).
   * **Comprehensive 13-Dataset Multi-Baseline Progression:** Multi-baseline comparisons across LIAR, CoAID, FEVER, AVeriTeC, FF++, Celeb-DF, ASVspoof, TwiBot-20, Cresci-17, etc., show systematic progression from TF-IDF ($0.5303$) to Full ARES 2.0 (**$\text{Macro-F1} = 0.9010$ [$95\%$ CI: $0.892, 0.910$]**) supported by paired bootstrap tests ($B=10,000$), exact McNemar tests, and Holm-Bonferroni corrections (**Table XXXVII**, Page 40).
   * **10-System Architectural Comparison Matrix:** Added **Table XXI** (Page 17) benchmarking ARES against 10 state-of-the-art verification systems across 8 functional dimensions.

5. **Full Public Open-Source Reproducibility:**
   * **GitHub Repository:** Publicly available under Apache-2.0 at `https://github.com/gurarpitzz/Ares-Cognitive-Computation` (Release Tag: `v2.0.0`).
   * **Permanent Zenodo Archive:** Complete codebase, models, and frozen dataset manifests archived with permanent DOI: `10.5281/zenodo.10826492`.
   * **Automated Unit Testing:** Complete test suite of **110 passing unit tests** executable via `pytest ares_v2/tests -q` in under 45 seconds.

6. **Comprehensive Citation Reconciliation:**
   Every single inline `[?]` marker has been resolved. The bibliography has been expanded to 65 verified peer-reviewed publications with active DOIs, and all citation brackets across the text, tables, and appendices have been rigorously verified.

---

### Accompanying Submission Files
1. **Main Manuscript:** `ARES_Cognitive_Computation_Revised.pdf` (Exactly 43 pages, standard double-column layout).
2. **Point-by-Point Response to Reviewers:** `ARES_response_to_reviewers.pdf` (Detailed, comprehensive rebuttal itemizing every critique and its exact section/page resolution).
3. **Open-Source Code & Data Manifests:** Fully accessible at the Zenodo and GitHub repositories referenced above.

### Mandatory Declarations
* **Originality & Prior Publication:** The manuscript represents original work that has not been published previously and is not under consideration for publication elsewhere.
* **Conflicts of Interest:** The author declares that there are no financial, commercial, institutional, or personal conflicts of interest that could influence the design, evaluation, or presentation of the findings reported in this manuscript.
* **Research Ethics & Human Subjects:** The study relies strictly on public benchmark datasets and synthetic perturbation simulations. No human subjects interventions or unauthorized operational deployments were conducted.
* **Open Access & Licensing:** The author confirms agreement with Springer Nature's publishing policies and Open Access licensing terms (CC BY 4.0).

We trust that the exhaustive improvements, empirical additions, and complete resolution of all previous concerns will allow the reviewers and editorial board to evaluate the scientific contributions of ARES on their true merits.

Thank you very much for your time, consideration, and continued editorial guidance.

Sincerely,

**Gurarpit Singh**  
Department of Computer Applications  
Vivekananda Institute of Professional Studies – Technical Campus (VIPS-TC)  
Guru Gobind Singh Indraprastha University (GGSIPU), Delhi, India  
Email: `bca_04817702024_gurarpit@vipstc.edu.in`  
ORCID: `0009-0002-8610-3885`  
Phone: +91 98118 20625  
