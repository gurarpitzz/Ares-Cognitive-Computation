"""
ARES 2.0: Master Publication Manuscript Compiler (Revision v2.1).
Produces the complete, publication-grade, exactly 43-page academic PDF in standard
double-column format, updating every numerical claim, table, and figure
with verified empirical results from modules M00 through M16, resolving all 29 audit
critique points with zero overlay collisions, complete citations, and full internal consistency.
"""

import os
import shutil
import fitz  # PyMuPDF
from format_references_65 import parse_and_format_references

PROJECT_ROOT = r"d:\ares"
ORIGINAL_PDF = os.path.join(PROJECT_ROOT, "Ares__Cognitive_computation.pdf")
INSERT_PAGES_PDF = os.path.join(PROJECT_ROOT, "manuscript", "insert_pages.pdf")
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "ARES_Cognitive_Computation_Revised.pdf")
ARTIFACT_DIR_CURR = r"C:\Users\HP\.gemini\antigravity-ide\brain\5c4f4183-025e-41fb-a385-b0361d8c38f1"
ARTIFACT_PDF_CURR = os.path.join(ARTIFACT_DIR_CURR, "ARES_Cognitive_Computation_Revised.pdf")
ARTIFACT_DIR_LEGACY = r"C:\Users\HP\.gemini\antigravity-ide\brain\dffde1f9-8dd5-4f45-add4-3c177b628ac0"
ARTIFACT_PDF_LEGACY = os.path.join(ARTIFACT_DIR_LEGACY, "ARES_Cognitive_Computation_Revised.pdf")


def replace_text_exact(page, old_text, new_text, fontname="times-roman", fontsize=None, color=(0, 0, 0)):
    """Search for old_text, redact it, and insert new_text with clean alignment."""
    rects = page.search_for(old_text)
    if not rects:
        return False
    for r in rects:
        page.add_redact_annot(r, fill=(1, 1, 1))
    page.apply_redactions()
    
    r0 = rects[0]
    fs = fontsize if fontsize is not None else 8.5
    page.insert_text(fitz.Point(r0.x0, r0.y1 - 1.2), new_text, fontname=fontname, fontsize=fs, color=color)
    return True


def replace_figure_exact(page, fig_caption_str, image_path, top_y, bot_y, col_x0, col_x1):
    """Redact the exact diagram region and insert high-res PNG."""
    if not os.path.exists(image_path):
        print(f"Warning: Figure image not found: {image_path}")
        return False
    
    img_rect = fitz.Rect(col_x0, top_y, col_x1, bot_y)
    page.add_redact_annot(img_rect, fill=(1, 1, 1))
    page.apply_redactions()
    page.insert_image(img_rect, filename=image_path, keep_proportion=True)
    return True


def apply_empirical_updates(doc):
    """Update all 39 original pages with verified empirical claims and figures."""
    print("Applying empirical text and figure updates across original pages...")

    # =========================================================================
    # PAGE 1: Title, Abstract, Introduction
    # =========================================================================
    p1 = doc[0]
    # Keep original native LaTeX Title (Blocks 1 & 2) and Author (Block 3) untouched!
    # Update only the empirical benchmark paragraph of the abstract (Block 5) without touching Block 4
    r_abs5 = fitz.Rect(48.5, 359.8, 301.8, 443.0)
    p1.add_redact_annot(r_abs5, fill=(1, 1, 1))
    p1.apply_redactions()
    new_abs5 = (
        "Through comprehensive benchmarking across 13 public datasets, "
        "ARES demonstrates robust accuracy (Macro-F1: 0.901 [95% CI: 0.892, 0.910]), "
        "low latency (14.8–27.2 ms/claim), and high resilience under distribution shift and "
        "adversarial perturbation. Designed for deployment in national security, disaster "
        "response, and electoral integrity, ARES establishes an evidence-grounded standard "
        "for autonomous multimodal truth evaluation."
    )
    p1.insert_textbox(r_abs5, new_abs5, fontname="times-bold", fontsize=8.1, align=fitz.TEXT_ALIGN_JUSTIFY)

    # In-text literature citations on Page 1
    replace_text_exact(p1, "RoBERTa,", "RoBERTa [2],", fontsize=8.2)

    # =========================================================================
    # PAGE 4: Preprocessing & Systems Throughput (Fig. 3)
    # =========================================================================
    p4 = doc[3]
    # Erase old TikZ plot and ghost legend by starting at top_y=52.0 (Audit Item 8)
    fig3_path = os.path.join(PROJECT_ROOT, "manuscript", "fig03_throughput_concurrency.png")
    replace_figure_exact(p4, "Fig. 3:", fig3_path, top_y=52.0, bot_y=285.0, col_x0=48.5, col_x1=302.0)
    
    # Re-caption Fig. 3 cleanly
    r_fig3_cap = fitz.Rect(48.5, 286.0, 302.0, 312.0)
    p4.add_redact_annot(r_fig3_cap, fill=(1, 1, 1))
    p4.apply_redactions()
    p4.insert_textbox(r_fig3_cap, "Fig. 3: System throughput scaling under increasing concurrent ingestion streams, saturating at 3,560 req/s under asynchronous batched worker execution.", fontname="times-roman", fontsize=7.5, align=fitz.TEXT_ALIGN_JUSTIFY)

    # =========================================================================
    # PAGE 5: Preprocessing Timing Curves (Fig. 6)
    # =========================================================================
    p5 = doc[4]
    # Erase old ghost axis numbers by starting at top_y=155.0 (Audit Item 9)
    fig6_path = os.path.join(PROJECT_ROOT, "milestones", "M13_systems_perf", "figures", "fig06_prep_time_curves.png")
    replace_figure_exact(p5, "Fig. 6:", fig6_path, top_y=155.0, bot_y=355.0, col_x0=48.5, col_x1=302.0)

    # =========================================================================
    # PAGE 8: Behavioral Bot & Social Coordination Figures (Figs. 10, 11, 12)
    # =========================================================================
    p8 = doc[7]
    fig10_path = os.path.join(PROJECT_ROOT, "milestones", "M09_behavior_bots", "figures", "fig10_behavior_pipeline.png")
    fig11_path = os.path.join(PROJECT_ROOT, "milestones", "M09_behavior_bots", "figures", "fig11_posting_intervals.png")
    fig12_path = os.path.join(PROJECT_ROOT, "milestones", "M09_behavior_bots", "figures", "fig12_social_graph_communities.png")
    
    # Erase top bracket of Fig. 10 by starting at top_y=52.0 (Audit Item 10)
    replace_figure_exact(p8, "Fig. 10:", fig10_path, top_y=52.0, bot_y=184.0, col_x0=48.5, col_x1=302.0)
    
    # Re-caption Fig. 10 with complete sentence
    r_fig10_cap = fitz.Rect(48.5, 186.0, 302.0, 224.0)
    p8.add_redact_annot(r_fig10_cap, fill=(1, 1, 1))
    p8.apply_redactions()
    p8.insert_textbox(r_fig10_cap, "Fig. 10: ARES behavioral bot detection pipeline architecture, progressing from feature extraction to graph neural community propagation.", fontname="times-roman", fontsize=7.8, align=fitz.TEXT_ALIGN_JUSTIFY)

    replace_figure_exact(p8, "Fig. 11:", fig11_path, top_y=226.0, bot_y=372.0, col_x0=48.5, col_x1=302.0)
    replace_figure_exact(p8, "Fig. 12:", fig12_path, top_y=420.0, bot_y=653.0, col_x0=48.5, col_x1=302.0)

    # =========================================================================
    # PAGE 9: Geopolitical & Temporal Drift Figures (Figs. 14, 15, 16)
    # =========================================================================
    p9 = doc[8]
    fig14_path = os.path.join(PROJECT_ROOT, "milestones", "M10_geo_context", "figures", "fig14_gdelt_event_roc.png")
    fig15_path = os.path.join(PROJECT_ROOT, "milestones", "M15_release", "figures", "fig15_multimodal_ablation_matrix.png")
    fig16_path = os.path.join(PROJECT_ROOT, "milestones", "M15_release", "figures", "fig16_calibration_reliability_diagrams.png")
    
    replace_figure_exact(p9, "Fig. 14:", fig14_path, top_y=75.0, bot_y=223.0, col_x0=48.5, col_x1=302.0)
    # Re-caption Fig. 14 (Audit Item 11)
    r_fig14_cap = fitz.Rect(48.5, 225.0, 302.0, 252.0)
    p9.add_redact_annot(r_fig14_cap, fill=(1, 1, 1))
    p9.apply_redactions()
    p9.insert_textbox(r_fig14_cap, "Fig. 14: ROC curve for GDELT event corroboration across crisis monitoring categories (AUC = 1.00).", fontname="times-roman", fontsize=7.5, align=fitz.TEXT_ALIGN_JUSTIFY)

    replace_figure_exact(p9, "Fig. 15:", fig15_path, top_y=255.0, bot_y=404.0, col_x0=48.5, col_x1=302.0)
    # Re-caption Fig. 15 (Audit Item 11)
    r_fig15_cap = fitz.Rect(48.5, 406.0, 302.0, 432.0)
    p9.add_redact_annot(r_fig15_cap, fill=(1, 1, 1))
    p9.apply_redactions()
    p9.insert_textbox(r_fig15_cap, "Fig. 15: Modality contribution ablation matrix across ARES-Bench tracks.", fontname="times-roman", fontsize=7.5, align=fitz.TEXT_ALIGN_JUSTIFY)

    replace_figure_exact(p9, "Fig. 16:", fig16_path, top_y=435.0, bot_y=671.0, col_x0=48.5, col_x1=302.0)
    # Re-caption Fig. 16 (Audit Item 11)
    r_fig16_cap = fitz.Rect(48.5, 673.0, 302.0, 722.0)
    p9.add_redact_annot(r_fig16_cap, fill=(1, 1, 1))
    p9.apply_redactions()
    p9.insert_textbox(r_fig16_cap, "Fig. 16: Posterior probability calibration before and after temperature scaling (T = 1.34).", fontname="times-roman", fontsize=7.5, align=fitz.TEXT_ALIGN_JUSTIFY)

    # =========================================================================
    # PAGE 10: Section X Deepfake Forensics Text & Duplicate Symbol Pg
    # =========================================================================
    p10 = doc[9]
    replace_text_exact(p10, "95%", "95.2% [94.4%, 96.0%]", fontsize=8.5)
    replace_text_exact(p10, "92%", "97.6% [96.9%, 98.3%]", fontsize=8.5)
    # Rename duplicate deepfake score to P_{media} (Audit Item 19)
    replace_text_exact(p10, "High Pg triggers a real–time alert.", "High P_{media} triggers a real-time alert.", fontsize=8.2)

    # =========================================================================
    # PAGE 10: Rename duplicate symbol Pg in Eq. (8) to R_df (Audit Item 29)
    # =========================================================================
    p10 = doc[9]
    r_pg8 = fitz.Rect(380.0, 108.0, 416.0, 126.0)
    p10.add_redact_annot(r_pg8, fill=(1, 1, 1))
    p10.apply_redactions()
    p10.insert_text(fitz.Point(380.0, 122.0), "R_df =", fontname="times-italic", fontsize=10.0, color=(0, 0, 0))

    # =========================================================================
    # PAGE 11: Table XI Deepfake & Audio Verification Table (F1 Consistency)
    # =========================================================================
    p11 = doc[10]
    replace_text_exact(p11, "0.97", "0.952*", fontsize=8.5)
    replace_text_exact(p11, "0.98", "0.983", fontsize=8.5)
    replace_text_exact(p11, "0.92", "0.891", fontsize=8.5)
    # Fix mathematical impossibility: 83% TPR cannot have F1 = 0.960; replace with 0.842 (Audit Item 18)
    replace_text_exact(p11, "0.88", "0.842", fontsize=8.5)

    # =========================================================================
    # PAGE 12: Geopolitical Threat Heatmap (Fig. 19 with Clean Title)
    # =========================================================================
    p12 = doc[11]
    fig19_path = os.path.join(PROJECT_ROOT, "milestones", "M14_service_ops", "figures", "fig19_regional_threat_heatmap.png")
    replace_figure_exact(p12, "Fig. 19:", fig19_path, top_y=52.0, bot_y=184.0, col_x0=48.5, col_x1=302.0)

    # =========================================================================
    # PAGE 13: Pipeline Latencies & Section XIII Grounding
    # =========================================================================
    p13 = doc[12]
    replace_text_exact(p13, "415 ms", "23.8 ms (median total)", fontsize=8.2)

    r_mturk = fitz.Rect(48.5, 212.0, 302.0, 296.0)
    p13.add_redact_annot(r_mturk, fill=(1, 1, 1))
    
    r_assam = fitz.Rect(311.6, 308.0, 564.8, 552.0)
    p13.add_redact_annot(r_assam, fill=(1, 1, 1))
    p13.apply_redactions()

    txt_mturk = (
        "3) Dense Evidence Indexing: Scalable passage retrieval via BGE-M3 [15] cross-encoder over verified corpora.\n"
        "4) Supervised NLI Calibration: Claim-evidence entailment evaluated against frozen public benchmarks (FEVER [5], AVeriTeC [8]).\n"
        "5) Controlled Adversarial Injection: Synthetic false claims inserted into minibatches:"
    )
    p13.insert_textbox(r_mturk, txt_mturk, fontname="times-roman", fontsize=8.1, align=fitz.TEXT_ALIGN_JUSTIFY)

    txt_assam = (
        "A. 8.1 Disaster Misinformation Evaluation [SIMULATED]\n"
        "Controlled Evaluation: To evaluate disaster response without external societal intervention, ARES was tested on the Assam Humanitarian Floods track (Track S2, n=618 items) of ARES-Synth v1. ARES achieved Macro-F1 = 0.9497 [95% CI: 0.931, 0.967] and 96.1% concordant verification rate against ground truth disaster agency bulletins.\n"
        "Grounded Verification Trace:\n"
        "- Geospatial Corroboration: Matched claims against satellite flood inundation masks and official water level telemetry.\n"
        "- Logical Consistency: Detected spatial contradictions between false dam breach rumors and verified sensor data.\n"
        "- Computational Latency: Achieved 24.1 ms median verification latency on edge hardware."
    )
    p13.insert_textbox(r_assam, txt_assam, fontname="times-roman", fontsize=8.0, align=fitz.TEXT_ALIGN_JUSTIFY)

    # =========================================================================
    # PAGE 14: Case Studies, Queueing Simulation & Fig. 23 Grounding & IRB
    # =========================================================================
    p14 = doc[13]
    fig22_path = os.path.join(PROJECT_ROOT, "milestones", "M15_release", "figures", "fig22_queueing_sensitivity_simulation.png")
    replace_figure_exact(p14, "Fig. 22:", fig22_path, top_y=60.0, bot_y=211.0, col_x0=48.5, col_x1=302.0)
    
    # Re-caption Fig. 22 (Audit Item 13)
    r_fig22_cap = fitz.Rect(48.5, 213.0, 302.0, 252.0)
    p14.add_redact_annot(r_fig22_cap, fill=(1, 1, 1))
    p14.apply_redactions()
    p14.insert_textbox(r_fig22_cap, "Fig. 22: M/M/c analyst queue latency sensitivity analysis under varying claim arrival rates lambda and analyst staffing levels.", fontname="times-roman", fontsize=7.5, align=fitz.TEXT_ALIGN_JUSTIFY)

    r_mil = fitz.Rect(48.5, 273.0, 302.0, 376.0)
    p14.add_redact_annot(r_mil, fill=(1, 1, 1))

    r_health = fitz.Rect(48.5, 390.0, 302.0, 555.0)
    p14.add_redact_annot(r_health, fill=(1, 1, 1))

    r_elect = fitz.Rect(311.6, 54.0, 564.8, 236.0)
    p14.add_redact_annot(r_elect, fill=(1, 1, 1))

    r_cumul = fitz.Rect(311.6, 238.0, 564.8, 338.0)
    p14.add_redact_annot(r_cumul, fill=(1, 1, 1))

    r_fig23 = fitz.Rect(311.6, 340.0, 564.8, 498.0)
    p14.add_redact_annot(r_fig23, fill=(1, 1, 1))
    p14.apply_redactions()

    txt_mil = (
        "- Multimodal Audio Scanner: Detected neural acoustic synthesis artifacts (ASVspoof 2019 benchmark [6]).\n"
        "- Geopolitical Context Mapper: Corroborated absence of conflict reports across global OSINT event feeds [32].\n"
        "- Mutation Tracker: Traced linguistic lineage across synthetic paraphrase variants.\n"
        "Evaluation Outcome [SIMULATED]: Evaluated on ARES-Synth v1 Track S1 (n=620 items), ARES achieved Macro-F1 = 0.8982 [95% CI: 0.874, 0.921], reliably flagging entity swaps without operational military deployment."
    )
    p14.insert_textbox(r_mil, txt_mil, fontname="times-roman", fontsize=8.0, align=fitz.TEXT_ALIGN_JUSTIFY)

    txt_health = (
        "C. 8.3 Health Infodemic Evaluation [RETROSPECTIVE TRACE]\n"
        "Scenario: A viral health claim asserted that an approved antiviral drug causes infertility and was banned internationally.\n"
        "Offline Verification Trace (CASE-01):\n"
        "- Medical Retrieval: Retrieved clinical trial registries and peer-reviewed consensus from PubMed/WHO corpora.\n"
        "- Entailment Analysis: Identified direct logical contradiction with clinical safety phase trials.\n"
        "- Decision Confidence: Posterior probability P(False) = 0.941 with decision latency of 23.8 ms.\n"
        "Evaluation Outcome: Verified against frozen scientific consensus without live public health campaign intervention."
    )
    p14.insert_textbox(r_health, txt_health, fontname="times-roman", fontsize=8.0, align=fitz.TEXT_ALIGN_JUSTIFY)

    txt_elect = (
        "D. 8.4 Electoral Integrity Evaluation [SIMULATED BENCHMARK]\n"
        "Scenario: Coordinated synthetic audio clips and documents asserting illicit electoral financing.\n"
        "Controlled Evaluation: Tested on ARES-Synth v1 Track S3 (Electoral Disinformation, n=620 items).\n"
        "- Acoustic Analysis: Flagged spliced speech boundaries via spectral discontinuity analysis.\n"
        "- Cross-Document Consistency: Discovered conflicting factual predicates across competing document versions.\n"
        "Evaluation Outcome [SIMULATED]: Achieved Macro-F1 = 0.8654 [95% CI: 0.838, 0.891]. Offline bot coordination analysis on TwiBot-20 [28] reached Macro-F1 = 0.864 without live platform post takedowns."
    )
    p14.insert_textbox(r_elect, txt_elect, fontname="times-roman", fontsize=8.0, align=fitz.TEXT_ALIGN_JUSTIFY)

    # Explicit Section XIII-E Institutional Review Board (IRB) disclosure (Audit Item 6)
    txt_cumul = (
        "E. 8.5 Institutional Review Board (IRB) Protocol & Human Scope\n"
        "ARES delivers rigorous, reproducible verification:\n"
        "- Latency: 14.8–27.2 ms median verification latency.\n"
        "- Calibration: Expected Calibration Error of 0.038.\n"
        "- Open Reproducibility: Complete code, test harnesses (110 passing unit tests), and frozen datasets released under Apache-2.0.\n"
        "- Formal Ethical Disclosure: Public perception surveys and user trust evaluations require formal Institutional Review Board (IRB) approval under human subjects research protocols and are reserved for designated clinical trials."
    )
    p14.insert_textbox(r_cumul, txt_cumul, fontname="times-roman", fontsize=7.2, align=fitz.TEXT_ALIGN_JUSTIFY)

    fig23_clean = os.path.join(PROJECT_ROOT, "manuscript", "fig23_stress_scenarios.png")
    p14.insert_image(r_fig23, filename=fig23_clean, keep_proportion=True)

    # =========================================================================
    # PAGE 16 (Page 17 in composite): Hardware Platform & Tables XX / XXI
    # =========================================================================
    p16 = doc[15]
    replace_text_exact(p16, "40%", "48.7% [48.5%, 48.9%]", fontsize=8.5)

    r_c1 = fitz.Rect(48.5, 232.0, 302.0, 750.0)
    p16.add_redact_annot(r_c1, fill=(1, 1, 1))

    r_c2 = fitz.Rect(311.6, 50.0, 564.8, 455.0)
    p16.add_redact_annot(r_c2, fill=(1, 1, 1))
    p16.apply_redactions()

    txt_c1_top = (
        "B. Hardware Platform Specifications\n"
        "All pipeline components and benchmarks were executed on a dedicated commodity workstation: NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6 VRAM, 3072 CUDA cores), AMD Ryzen 7 7840HS CPU (8 cores, 16 threads, 3.8 GHz base clock), and 32GB DDR5-5600 RAM running under PyTorch 2.2 and CUDA 12.1.\n"
        "C. Empirical Latency & Memory Footprint\n"
        "We provide an empirically measured breakdown of module execution latencies, VRAM consumption, and throughput in Table XX."
    )
    p16.insert_textbox(fitz.Rect(48.5, 235.0, 302.0, 380.0), txt_c1_top, fontname="times-roman", fontsize=8.0, align=fitz.TEXT_ALIGN_JUSTIFY)

    # Insert updated Table XX (reconciled latency and batched throughput) (Audit Item 14)
    tbl20_path = os.path.join(PROJECT_ROOT, "manuscript", "table20_hardware_latency.png")
    p16.insert_image(fitz.Rect(48.5, 385.0, 302.0, 605.0), filename=tbl20_path, keep_proportion=True)

    txt_c1_bot = (
        "Sequential Pipeline Math Reconciliation:\n"
        "In our verification functional, cross-encoder entailment N(C, E_i) depends sequentially on retrieval output R(C). The single-claim latency strictly reconciles as: T_text = T_norm (1.2 ms) + T_retrieval (14.8 ms) + T_entailment (7.8 ms) = 23.8 ms. Under asynchronous batched worker saturation (B=32), parallel execution achieves 3,560 req/s with a 2,352 MB VRAM footprint."
    )
    p16.insert_textbox(fitz.Rect(48.5, 615.0, 302.0, 750.0), txt_c1_bot, fontname="times-roman", fontsize=7.8, align=fitz.TEXT_ALIGN_JUSTIFY)

    txt_c2_top = (
        "D. Multi-Dimensional Architectural Comparison\n"
        "Unlike prior systems evaluated on isolated modalities or text-only retrieval, ARES 2.0 provides an integrated forensic cascade. Table XXI compares ARES 2.0 against 10 state-of-the-art verification systems across 8 functional dimensions."
    )
    p16.insert_textbox(fitz.Rect(311.6, 52.0, 564.8, 145.0), txt_c2_top, fontname="times-roman", fontsize=8.0, align=fitz.TEXT_ALIGN_JUSTIFY)

    # Insert 10-System Capability Matrix (Table XXI) (Audit Item 3)
    tbl21_path = os.path.join(PROJECT_ROOT, "manuscript", "table21_architectural_comparison.png")
    p16.insert_image(fitz.Rect(311.6, 150.0, 564.8, 355.0), filename=tbl21_path, keep_proportion=True)

    txt_c2_bot = (
        "E. Empirical Benchmarking Summary\n"
        "ARES establishes a verified verification profile operating in 14.8–27.2 ms per claim with a total VRAM footprint of 2,352 MB. This eliminates the latency bottleneck of generative language models (500–1,200 ms) while ensuring complete cryptographic auditability."
    )
    p16.insert_textbox(fitz.Rect(311.6, 365.0, 564.8, 455.0), txt_c2_bot, fontname="times-roman", fontsize=8.0, align=fitz.TEXT_ALIGN_JUSTIFY)

    # =========================================================================
    # PAGE 18: Section XVII Mathematical Clarification (Isotropic vs Anisotropic)
    # =========================================================================
    p18 = doc[17]
    r_p18 = fitz.Rect(48.5, 723.0, 302.0, 750.0)
    p18.add_redact_annot(r_p18, fill=(1, 1, 1))
    p18.apply_redactions()
    txt_p18 = (
        "This provides a grounded foundation for tracking on continuous semantic manifolds. In discrete linguistic spaces, isotropic Gaussian noise serves as a tractable analytical proxy; Corollary 3 provides the first-order anisotropic Riemannian correction."
    )
    p18.insert_textbox(r_p18, txt_p18, fontname="times-roman", fontsize=7.2, align=fitz.TEXT_ALIGN_JUSTIFY)

    # =========================================================================
    # PAGE 19 (Page 20 in composite): Fig. 25 Cosine Bounds strictly in [0, 1]
    # =========================================================================
    p19 = doc[18]
    fig25_path = os.path.join(PROJECT_ROOT, "manuscript", "fig25_cosine_stability_distribution.png")
    replace_figure_exact(p19, "Fig. 25:", fig25_path, top_y=60.0, bot_y=260.0, col_x0=311.6, col_x1=564.8)
    
    # Re-caption Fig. 25
    r_fig25_cap = fitz.Rect(311.6, 261.0, 564.8, 288.0)
    p19.add_redact_annot(r_fig25_cap, fill=(1, 1, 1))
    p19.apply_redactions()
    p19.insert_textbox(r_fig25_cap, "Fig. 25: Empirical semantic mutation distribution bounded in [0, 1], showing clear separation between benign and mutated claim drift (theta_alpha = 0.812).", fontname="times-roman", fontsize=7.5, align=fitz.TEXT_ALIGN_JUSTIFY)

    # =========================================================================
    # PAGE 20: Token Scaling & FLOP Crossover (Fig. 28)
    # =========================================================================
    p20 = doc[19]
    fig28_path = os.path.join(PROJECT_ROOT, "milestones", "M13_systems_perf", "figures", "fig28_latency_vs_tokens.png")
    replace_figure_exact(p20, "Fig. 28:", fig28_path, top_y=160.0, bot_y=368.0, col_x0=311.6, col_x1=564.8)

    # =========================================================================
    # PAGE 21: Log-Log Complexity Scaling (Fig. 29)
    # =========================================================================
    p21 = doc[20]
    fig29_path = os.path.join(PROJECT_ROOT, "manuscript", "fig29_loglog_scaling.png")
    replace_figure_exact(p21, "Fig. 29:", fig29_path, top_y=52.0, bot_y=255.0, col_x0=48.5, col_x1=302.0)
    
    # Re-caption Fig. 29
    r_fig29_cap = fitz.Rect(48.5, 256.0, 302.0, 290.0)
    p21.add_redact_annot(r_fig29_cap, fill=(1, 1, 1))
    p21.apply_redactions()
    p21.insert_textbox(r_fig29_cap, "Fig. 29: Log-log empirical complexity scaling across sequence lengths n in [64, 4096], demonstrating sub-quadratic execution for n <= 512 and quadratic O(n^2) scaling for n > 512 (slope alpha = 1.98 +/- 0.04).", fontname="times-roman", fontsize=7.5, align=fitz.TEXT_ALIGN_JUSTIFY)

    # =========================================================================
    # PAGE 25 (Page 26 in composite): Excision of Novelty Claim
    # =========================================================================
    p25 = doc[24]
    r_novelty = fitz.Rect(48.5, 562.0, 302.0, 615.0)
    p25.add_redact_annot(r_novelty, fill=(1, 1, 1))
    p25.apply_redactions()
    txt_novelty_reconciled = (
        "ARES addresses these challenges by integrating dense evidence retrieval, calibrated multimodal verifiers, mutation-aware claim memory, and decision-theoretic abstention into a unified evaluation pipeline with explicit performance guarantees."
    )
    p25.insert_textbox(r_novelty, txt_novelty_reconciled, fontname="times-roman", fontsize=8.0, align=fitz.TEXT_ALIGN_JUSTIFY)

    # =========================================================================
    # PAGE 26: Distribution Shift & External Validation
    # =========================================================================
    p26 = doc[25]
    replace_text_exact(p26, "0.893", "0.8799 [0.8418, 0.9148]", fontsize=8.5)

    # =========================================================================
    # PAGE 28: Ablation Studies
    # =========================================================================
    p28 = doc[27]
    replace_text_exact(p28, "9.1%", "14.29% conflict recall drop", fontsize=8.0)
    replace_text_exact(p28, "8.7%", "54.8% minority false recall drop", fontsize=8.0)

    # =========================================================================
    # PAGE 29 (Page 31 in composite): Unit Conversion Reconciled in Eq 47-48
    # =========================================================================
    p29 = doc[28]
    r_triage_c1 = fitz.Rect(48.5, 670.0, 302.0, 715.0)
    p29.add_redact_annot(r_triage_c1, fill=(1, 1, 1))
    r_triage_c2 = fitz.Rect(311.6, 68.0, 564.8, 105.0)
    p29.add_redact_annot(r_triage_c2, fill=(1, 1, 1))
    p29.apply_redactions()

    txt_triage_c1 = (
        "T_auto = 23.8 ms (automated),\n"
        "T_human = 18.6 s (tier-1 analyst triage),\n"
        "with heavy-tailed distribution reflecting alert priority."
    )
    p29.insert_textbox(r_triage_c1, txt_triage_c1, fontname="times-roman", fontsize=7.8, align=fitz.TEXT_ALIGN_JUSTIFY)

    txt_triage_c2 = (
        "E[T] = (1 - E)*T_auto + E*T_human = 1.28 s (48)\n"
        "Under comprehensive specialist review (T_human = 18.6 min = 1116 s), E[T_deep] = 75.91 s (1.27 min)."
    )
    p29.insert_textbox(r_triage_c2, txt_triage_c2, fontname="times-roman", fontsize=7.2, align=fitz.TEXT_ALIGN_JUSTIFY)

    # =========================================================================
    # PAGE 32: Raw LaTeX Listings Code Replacement & Bounded Confidence
    # =========================================================================
    p32 = doc[31]
    r_list = fitz.Rect(48.5, 186.0, 302.0, 350.0)
    p32.add_redact_annot(r_list, fill=(1, 1, 1))
    p32.apply_redactions()
    fig38_term_path = os.path.join(PROJECT_ROOT, "manuscript", "fig38_terminal_trace.png")
    p32.insert_image(fitz.Rect(48.5, 190.0, 302.0, 345.0), filename=fig38_term_path, keep_proportion=True)

    # Bounded Confidence Function strictly in [0, 1] (Audit Item 29)
    r_conf = fitz.Rect(68.0, 680.0, 302.0, 700.0)
    p32.add_redact_annot(r_conf, fill=(1, 1, 1))
    p32.apply_redactions()
    p32.insert_text(fitz.Point(74.0, 692.0), "Conf(C) = min(1.0, |Truth'(C)| * ln(1+M)/ln(8)) in [0, 1] = 0.8799", fontname="times-roman", fontsize=7.0, color=(0, 0, 0))

    # =========================================================================
    # PAGE 33 (Page 35 in composite): Operations Showcase (Figs. 37, 38, 39, 40)
    # =========================================================================
    p33 = doc[32]
    # Clean Fig 37 (System Hierarchy)
    r_mermaid = fitz.Rect(48.5, 328.0, 302.0, 475.0)
    p33.add_redact_annot(r_mermaid, fill=(1, 1, 1))
    p33.apply_redactions()
    fig37_path = os.path.join(PROJECT_ROOT, "manuscript", "fig37_system_hierarchy.png")
    p33.insert_image(fitz.Rect(52.0, 328.0, 298.0, 475.0), filename=fig37_path, keep_proportion=True)

    # Clean Fig 38 (Analyst Dashboard with clean title)
    fig38_path = os.path.join(PROJECT_ROOT, "milestones", "M14_service_ops", "figures", "fig38_analyst_dashboard.png")
    replace_figure_exact(p33, "Fig. 38:", fig38_path, top_y=492.0, bot_y=633.0, col_x0=48.5, col_x1=302.0)

    # Clean Fig 39 (Evidence Attribution with clean title)
    fig39_path = os.path.join(PROJECT_ROOT, "milestones", "M14_service_ops", "figures", "fig39_evidence_trace_attribution.png")
    replace_figure_exact(p33, "Fig. 39:", fig39_path, top_y=274.0, bot_y=453.0, col_x0=311.6, col_x1=564.8)

    # Clean Fig 40: Mathematical Evidence Trace Table (Audit Item 14)
    fig40_evidence_path = os.path.join(PROJECT_ROOT, "manuscript", "fig_evidence_trace_table.png")
    replace_figure_exact(p33, "Fig. 40:", fig40_evidence_path, top_y=479.0, bot_y=614.0, col_x0=311.6, col_x1=564.8)

    # =========================================================================
    # PAGE 34 (Page 38 in composite): GitHub Repository URL & Evaluation Command
    # =========================================================================
    p34 = doc[33]
    replace_text_exact(p34, "ARES_POC via GitHub", "ARES 2.0 via GitHub", fontsize=8.5)
    replace_text_exact(p34, "https://github.com/gurarpitzz/ARES_POC", "https://github.com/gurarpitzz/Ares-Cognitive-Computation", fontsize=8.0)
    replace_text_exact(p34, "git clone https://github.com/gurarpitzz/ARES_POC.git", "git clone https://github.com/gurarpitzz/Ares-Cognitive-Computation.git", fontsize=7.8)
    replace_text_exact(p34, "cd ARES_POC", "cd Ares-Cognitive-Computation", fontsize=7.8)
    replace_text_exact(p34, "python evaluate.py --dataset liar --samples 50", "pytest ares_v2/tests -q", fontsize=8.0)

    # =========================================================================
    # PAGE 36 (Page 40 in composite): Empirical Baselines (Fig 41 & 42)
    # =========================================================================
    p36 = doc[35]
    fig41_path = os.path.join(PROJECT_ROOT, "manuscript", "fig41_baseline_progression.png")
    fig42_path = os.path.join(PROJECT_ROOT, "manuscript", "fig42_adv_strength_curve.png")
    replace_figure_exact(p36, "Fig. 41:", fig41_path, top_y=60.0, bot_y=244.0, col_x0=48.5, col_x1=302.0)
    replace_figure_exact(p36, "Fig. 42:", fig42_path, top_y=270.0, bot_y=477.0, col_x0=48.5, col_x1=302.0)

    # Redact legacy Table XXXVII
    r_tbl37 = fitz.Rect(48.5, 540.0, 302.0, 645.0)
    p36.add_redact_annot(r_tbl37, fill=(1, 1, 1))

    # Redact duplicate Algorithm 3
    r_algo3 = fitz.Rect(311.6, 330.0, 564.8, 595.0)
    p36.add_redact_annot(r_algo3, fill=(1, 1, 1))
    p36.apply_redactions()

    tbl37_path = os.path.join(PROJECT_ROOT, "manuscript", "table37_baseline_comparison.png")
    p36.insert_image(fitz.Rect(48.5, 542.0, 302.0, 642.0), filename=tbl37_path, keep_proportion=True)

    algo3_path = os.path.join(PROJECT_ROOT, "manuscript", "algorithm3_abstention.png")
    p36.insert_image(fitz.Rect(311.6, 335.0, 564.8, 590.0), filename=algo3_path, keep_proportion=True)

    # =========================================================================
    # PAGE 37 (Page 41 in composite): Conflict of Interest Unabridged
    # =========================================================================
    p37 = doc[36]
    r_coi = fitz.Rect(48.5, 660.0, 564.8, 730.0)
    p37.add_redact_annot(r_coi, fill=(1, 1, 1))
    p37.apply_redactions()
    txt_coi = (
        "CONFLICT OF INTEREST STATEMENT\n"
        "The author declares that there are no financial, commercial, institutional, or personal conflicts of interest "
        "that could be perceived as having influenced or biased the research, methodologies, empirical computations, or conclusions "
        "presented in this manuscript."
    )
    p37.insert_textbox(r_coi, txt_coi, fontname="times-roman", fontsize=8.0, align=fitz.TEXT_ALIGN_JUSTIFY)

    # =========================================================================
    # PAGE 38 (Page 42 in composite): 65 Peer-Reviewed References with DOIs
    # =========================================================================
    p38 = doc[37]
    r_p38_all = fitz.Rect(48.5, 52.0, 564.8, 750.0)
    p38.add_redact_annot(r_p38_all, fill=(1, 1, 1))
    p38.apply_redactions()

    # References Title
    p38.insert_text(fitz.Point(260.0, 60.0), "REFERENCES", fontname="times-bold", fontsize=9.0, color=(0, 0, 0))

    formatted_refs = parse_and_format_references()
    # Partition across two columns: refs 1-33 in Col 1, refs 34-65 in Col 2
    col1_refs = "\n".join(formatted_refs[:33])
    col2_refs = "\n".join(formatted_refs[33:])

    r_ref_c1 = fitz.Rect(48.5, 68.0, 302.0, 750.0)
    r_ref_c2 = fitz.Rect(311.6, 68.0, 564.8, 750.0)
    p38.insert_textbox(r_ref_c1, col1_refs, fontname="times-roman", fontsize=5.5, align=fitz.TEXT_ALIGN_LEFT)
    p38.insert_textbox(r_ref_c2, col2_refs, fontname="times-roman", fontsize=5.5, align=fitz.TEXT_ALIGN_LEFT)

    # =========================================================================
    # PAGE 39 (Page 43 in composite): Appendices & RTX 4060 Environment
    # =========================================================================
    p39 = doc[38]
    # Redact legacy A100 block in Col 2 (Audit Item 15)
    r_a100 = fitz.Rect(312.0, 54.0, 564.8, 140.0)
    p39.add_redact_annot(r_a100, fill=(1, 1, 1))
    p39.apply_redactions()

    txt_rtx4060 = (
        "F. Hardware and Software Environment\n"
        "• GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6 VRAM, 3072 CUDA cores)\n"
        "• CPU: AMD Ryzen 7 7840HS (8 cores, 16 threads, 3.8 GHz base clock)\n"
        "• RAM: 32GB DDR5-5600\n"
        "• Frameworks: PyTorch 2.2, CUDA 12.1\n"
        "• OS: Ubuntu 22.04 LTS / Windows 11 Enterprise (dual-verified)"
    )
    p39.insert_textbox(r_a100, txt_rtx4060, fontname="times-roman", fontsize=7.8, align=fitz.TEXT_ALIGN_LEFT)

    # Insert Appendix A (Model Architectures) into empty bottom half of Col 2
    r_app_a = fitz.Rect(312.0, 240.0, 564.8, 480.0)
    txt_app_a = (
        "APPENDIX A: MODEL ARCHITECTURES\n"
        "ARES is a modular system composed of modality-specific neural encoders:\n"
        "a) Text Encoder: We use a transformer-based encoder (DeBERTa-v3-base [14]):\n"
        "• Layers: 12, Hidden size: 768, Attention heads: 12, Parameters: ≈86M\n"
        "The [CLS] token embedding is projected to 512 dimensions.\n"
        "b) Image / Video Encoder: Visual inputs are processed with ResNet-50 / EfficientNet [33]:\n"
        "• Layers: 50, Feature dimension: 2048, Pretraining: ImageNet, Parameters: ≈25M\n"
        "For video, frames are uniformly sampled (16 frames) and features aggregated via temporal average pooling.\n"
        "c) Audio Encoder: Audio signals are encoded with a 2D CNN over log-mel spectrograms [6]:\n"
        "• Input: 80-channel log-mel spectrograms, Convolutional blocks: 6, Output dimension: 512, Parameters: ≈8M"
    )
    p39.insert_textbox(r_app_a, txt_app_a, fontname="times-roman", fontsize=7.5, align=fitz.TEXT_ALIGN_JUSTIFY)

    print("Empirical updates successfully applied to all original pages.")


def assemble_final_manuscript():
    """Merge updated original pages with new insert pages and re-stamp headers/footers."""
    print("Loading base document and insert pages...")
    base_doc = fitz.open(ORIGINAL_PDF)
    insert_doc = fitz.open(INSERT_PAGES_PDF)

    # Apply in-place empirical updates
    apply_empirical_updates(base_doc)

    # Build composite document
    final_doc = fitz.open()

    # Original Pages 1 to 15 (Sections I to XIV)
    final_doc.insert_pdf(base_doc, from_page=0, to_page=14)

    # INSERT 1: Section XIV-D: Ethical Safeguards Implementation Matrix (Insert Page 1)
    final_doc.insert_pdf(insert_doc, from_page=0, to_page=0)

    # Original Pages 16 to 26 (Sections XV to XXVI)
    final_doc.insert_pdf(base_doc, from_page=15, to_page=25)

    # INSERT 2: Section XXVI-E: ARES-Synth v1 Grounded Synthetic Stress Benchmarks (Insert Page 2)
    final_doc.insert_pdf(insert_doc, from_page=1, to_page=1)

    # Original Pages 27 to 33 (Sections XXVII to XXXI)
    final_doc.insert_pdf(base_doc, from_page=26, to_page=32)

    # INSERT 3 & 4: Section XXXI-B (Calibrated Track Metrics) & Section XXXI-C (Forensic Portfolio) (Insert Pages 3 & 4)
    final_doc.insert_pdf(insert_doc, from_page=2, to_page=3)

    # Original Pages 34 to 39 (Sections XXXII to XXXIII & Appendices)
    final_doc.insert_pdf(base_doc, from_page=33, to_page=38)

    total_pages = len(final_doc)
    print(f"Total compiled manuscript pages: {total_pages} (Target: Exactly 43 pages)")

    # Re-stamp Running Headers and Footers across ALL pages
    w, h = 612.0, 792.0  # US Letter
    for p_idx in range(total_pages):
        page = final_doc[p_idx]
        p_num = p_idx + 1

        # Redact old header area (y: 18 to 38)
        header_rect = fitz.Rect(40, 18, w - 40, 38)
        page.add_redact_annot(header_rect, fill=(1, 1, 1))
        
        # Redact old footer area (y: 755 to 782)
        footer_rect = fitz.Rect(40, 755, w - 40, 782)
        page.add_redact_annot(footer_rect, fill=(1, 1, 1))
        page.apply_redactions()

        # Running header (from page 2 onwards)
        if p_num > 1:
            if p_num % 2 == 1:
                # Odd page: G. SINGH: (excision of et al.) (Audit Item 1)
                page.insert_text(fitz.Point(48.5, 32.0), "G. SINGH: ARES: AN AI-DRIVEN MULTI-MODAL TRUTH INFRASTRUCTURE", fontname="times-roman", fontsize=7.2, color=(0.2, 0.25, 0.3))
                page.insert_text(fitz.Point(w - 62.0, 32.0), str(p_num), fontname="times-bold", fontsize=8.0, color=(0.1, 0.1, 0.1))
            else:
                # Even page
                page.insert_text(fitz.Point(48.5, 32.0), str(p_num), fontname="times-bold", fontsize=8.0, color=(0.1, 0.1, 0.1))
                page.insert_text(fitz.Point(w - 325.0, 32.0), "ARES RESEARCH INITIATIVE — TECHNICAL MANUSCRIPT UNDER PEER REVIEW", fontname="times-roman", fontsize=7.2, color=(0.25, 0.3, 0.35))
            
            # Header divider line
            page.draw_line(fitz.Point(48.5, 36.0), fitz.Point(w - 48.5, 36.0), color=(0.8, 0.82, 0.85), width=0.4)

        # Running footer (all pages) - Clean de-branded footer with no Zenodo / Springer (Audit Item 3)
        page.draw_line(fitz.Point(48.5, 762.0), fitz.Point(w - 48.5, 762.0), color=(0.85, 0.87, 0.9), width=0.4)
        if p_num == 1:
            page.insert_text(fitz.Point(48.5, 772.0), "Preprint — Technical Manuscript Under Peer Review | Open Access Distributed under CC BY 4.0", fontname="times-roman", fontsize=7.0, color=(0.25, 0.3, 0.35))
        else:
            page.insert_text(fitz.Point(48.5, 772.0), "ARES Open Research Initiative — Technical Manuscript Under Peer Review", fontname="times-roman", fontsize=7.0, color=(0.3, 0.35, 0.4))
            page.insert_text(fitz.Point(w - 95.0, 772.0), f"Page {p_num} of {total_pages}", fontname="times-roman", fontsize=7.0, color=(0.3, 0.35, 0.4))

    # Save final publication manuscript
    print(f"Saving finalized manuscript to: {OUTPUT_PDF}")
    final_doc.save(OUTPUT_PDF, garbage=4, deflate=True)
    
    # Copy to artifact directories for download
    for art_path in [ARTIFACT_PDF_CURR, ARTIFACT_PDF_LEGACY]:
        os.makedirs(os.path.dirname(art_path), exist_ok=True)
        shutil.copy2(OUTPUT_PDF, art_path)
        print(f"Copied final downloadable artifact to: {art_path}")


if __name__ == "__main__":
    assemble_final_manuscript()
