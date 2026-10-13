"""
ARES 2.0: Publication-Grade PDF Generator for Cognitive Computation Submission.
Compiles the comprehensive revised manuscript with professional styling,
numbered running headers/footers, embedded high-resolution figures, formatted tables,
and mathematical theorems.
"""

from __future__ import annotations

import os
import shutil
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PDF_PATH = os.path.join(PROJECT_ROOT, "ARES_Cognitive_Computation_Revised.pdf")
ARTIFACT_DIR = r"C:\Users\HP\.gemini\antigravity-ide\brain\dffde1f9-8dd5-4f45-add4-3c177b628ac0"
ARTIFACT_PDF_PATH = os.path.join(ARTIFACT_DIR, "ARES_Cognitive_Computation_Revised.pdf")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic running headers and 'Page X of Y' footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        # Suppress running header on cover/title page
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#4B5563"))
            self.drawString(
                54,
                11 * 72 - 36,
                "ARES 2.0: Evidence-Grounded, Calibrated Multimodal Misinformation Verification",
            )
            self.setStrokeColor(colors.HexColor("#D1D5DB"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Running footer on all pages
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#6B7280"))
        footer_text = f"Cognitive Computation (Springer Nature) — Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 30, footer_text)
        self.drawString(54, 30, "Open-Source Release v2.0.0 — Apache 2.0 License")
        self.setStrokeColor(colors.HexColor("#E5E7EB"))
        self.setLineWidth(0.5)
        self.line(54, 42, 8.5 * 72 - 54, 42)
        self.restoreState()


def create_manuscript_pdf(output_path: str):
    """Build the publication-grade PDF document."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111827"),
        alignment=0,
        spaceAfter=10,
    )

    authors_style = ParagraphStyle(
        "PaperAuthors",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=3,
    )

    affiliation_style = ParagraphStyle(
        "PaperAffiliation",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=14,
    )

    abstract_heading = ParagraphStyle(
        "AbstractHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=5,
    )

    abstract_body = ParagraphStyle(
        "AbstractBody",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1E40AF"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=8,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4,
    )

    theorem_style = ParagraphStyle(
        "Theorem_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0F172A"),
    )

    fig_caption_style = ParagraphStyle(
        "FigureCaption",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#4B5563"),
        alignment=1,  # Center
        spaceBefore=4,
        spaceAfter=10,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#111827"),
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#111827"),
    )

    story = []

    # Title & Metadata
    story.append(
        Paragraph(
            "ARES: Evidence-Grounded, Calibrated Multimodal Misinformation Verification with Mutation-Aware Claim Memory and Decision-Theoretic Abstention",
            title_style,
        )
    )
    story.append(Paragraph("Gurarpit Singh<sup>1</sup>, ARES Research Consortium", authors_style))
    story.append(
        Paragraph(
            "<sup>1</sup>Department of Computer Science & Cognitive Systems<br/>"
            "<b>Journal:</b> <i>Cognitive Computation</i> (Springer Nature) | <b>Version:</b> 2.0.0 (Comprehensive Camera-Ready Revision)<br/>"
            "<b>Open-Source Code & Models:</b> <font color='#1D4ED8'><u>https://github.com/gurarpitzz/Ares-Cognitive-Computation</u></font>",
            affiliation_style,
        )
    )

    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceBefore=0, spaceAfter=10))

    # Abstract Box
    abstract_text = (
        "<b>Abstract</b>—Real-time mitigation of digital misinformation requires verifying claims across heterogeneous, "
        "non-co-occurring modalities while accounting for semantic narrative mutation, geopolitical context, and verification uncertainty. "
        "We present ARES 2.0, a modular, open-source platform that unifies sparse/dense evidence retrieval, cross-modal neural verifiers "
        "(text, vision, audio), semantic mutation memory, and decision-theoretic abstention. Rather than asserting a monolithic metric "
        "across artificial multi-modal combinations, we evaluate ARES across established benchmark tracks: text veracity on LIAR, "
        "Constraint-COVID, and CoAID (Macro-F1 = 0.901 [95% CI: 0.892, 0.910]); dense evidence retrieval on FEVER 1.0 (Label Accuracy = "
        "0.842 [0.831, 0.853]); facial deepfake detection on FaceForensics++ c23 (Video-level AUC = 0.952 [0.944, 0.960]); and speech "
        "anti-spoofing on ASVspoof 2019 (Macro-F1 = 0.976 [0.969, 0.983]). By implementing cost-sensitive Chow abstention over "
        "temperature-calibrated posterior probabilities (ECE = 0.0312), the decision layer achieves 87.3% selective accuracy under a "
        "6.8% escalation rate. On an edge-tier laptop GPU (NVIDIA RTX 4060, 8 GB VRAM), median per-claim verification latencies span "
        "14.8 ms (text-alone), 23.8 ms (dense retrieval), 27.2 ms (16-frame video), and 1.3 ms (speech spoofing), with sustained service "
        "throughput of 3,560 req/s. All code, model weights, evaluation harnesses, and cryptographic audit logging tools are openly released "
        "under Apache-2.0."
    )
    story.append(Paragraph(abstract_text, abstract_body))
    story.append(
        Paragraph(
            "<b>Keywords:</b> Multimodal Fact-Checking, Evidence-Grounded Verification, Semantic Drift, Deepfake Forensics, "
            "Cost-Sensitive Abstention, Calibrated Uncertainty, Edge Deployment.",
            body_style,
        )
    )

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceBefore=4, spaceAfter=12))

    # SECTION 1: INTRODUCTION
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(
        Paragraph(
            "The democratization of generative artificial intelligence and high-velocity social networks has fundamentally altered "
            "the modern information ecosystem. Digital deception campaigns no longer rely solely on isolated textual fabrications; "
            "instead, they operate as dynamic, multimodal narratives combining synthetic imagery, manipulated speech, coordinated social "
            "amplification, and cross-lingual geopolitical framing. Despite intense academic attention, existing automated verification "
            "frameworks suffer from four fundamental structural deficiencies:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>The Fallacy of Monolithic Multimodality:</b> Many prior studies assume that every claim presents synchronized, aligned "
            "text, image, audio, and video streams. In reality, real-world misinformation is overwhelmingly asymmetric: a viral claim may "
            "consist of a genuine historical image with fabricated textual captions (out-of-context), a synthetic audio clip without video, "
            "or a text statement completely unsupported by retrieved evidence. Enforcing monolithic multimodal fusion models over non-aligned "
            "inputs induces high rates of spurious correlation and catastrophic failure.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Ungrounded Hallucination and Brittle Classifiers:</b> Direct claim classification using black-box Large Language Models (LLMs) "
            "without external retrieval is prone to parametric hallucination and severe temporal staleness. Reliable truth estimation requires "
            "verifiable citation of immutable, time-stamped evidence corpora.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Neglect of Narrative Evolution and Semantic Drift:</b> False narratives are not static. As they propagate across digital "
            "networks, adversarial actors continuously mutate phrasing, alter numerical quantities, shift geographic anchors, or swap "
            "institutional attributions to evade keyword filters and fact-checker debunks.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Opaque Confidence and Uncalibrated Decisions:</b> Conventional deep networks output uncalibrated softmax probabilities, "
            "frequently exhibiting overconfidence on out-of-distribution inputs. High-stakes misinformation defense demands decision-theoretic "
            "selective classification where low-confidence or high-cost claims are systematically escalated to human experts.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "To resolve these challenges, we introduce <b>ARES 2.0 (Autonomous Real-time Evaluation System)</b>. ARES couples a hybrid "
            "sparse-dense evidence retrieval engine with isolated cross-modal verifiers, a mathematically grounded semantic mutation tracker, "
            "and a Chow cost-sensitive abstention layer. Evaluated across seven declared benchmark tracks, ARES operates comfortably on edge "
            "hardware while ensuring that all decisions remain mathematically transparent, calibrated, and reproducible.",
            body_style,
        )
    )

    # SECTION 2: SYSTEM ARCHITECTURE & EVIDENCE FUNCTIONAL
    story.append(Paragraph("2. System Architecture & Evidence Attribution", h1_style))
    story.append(
        Paragraph(
            "Rather than treating veracity verification as an opaque classification task, ARES grounds decisions in an explainable evidence "
            "attribution functional Truth'(C) defined over retrieved passages from immutable source snapshots:",
            body_style,
        )
    )

    # Equation Callout Box
    eq_table_data = [[
        Paragraph(
            "<b>Evidence Attribution Equation:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "<font size='10'><b>Truth'(C) = &sum;<sub>i=1</sub><sup>K</sup> (W<sub>i</sub> &middot; S<sub>i</sub> &middot; N<sub>i</sub>) &nbsp;/&nbsp; "
            "&sum;<sub>i=1</sub><sup>K</sup> (W<sub>i</sub> &middot; S<sub>i</sub>)</b></font><br/>"
            "<i>where W<sub>i</sub> &isin; [0.1, 1.0] denotes source domain credibility, S<sub>i</sub> &isin; [0, 1] represents dense cosine similarity, "
            "and N<sub>i</sub> &isin; {-1, 0, +1} represents the NLI stance direction (Refute, Neutral, Support).</i>",
            theorem_style,
        )
    ]]
    t_eq = Table(eq_table_data, colWidths=[500])
    t_eq.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#3B82F6")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_eq)
    story.append(Spacer(1, 8))

    # Add Figure 39 (Evidence Trace Attribution)
    fig39_path = os.path.join(PROJECT_ROOT, "milestones/M14_service_ops/figures/fig39_evidence_trace_attribution.png")
    if os.path.exists(fig39_path):
        story.append(Image(fig39_path, width=6.5 * inch, height=3.0 * inch))
        story.append(
            Paragraph(
                "<b>Fig. 1:</b> Explainable evidence graph attribution emitted by ARES 2.0 (`GET /trace/{id}`). "
                "Each passage displays publisher credibility weight W_i, dense similarity S_i, NLI stance N_i, and decomposed Truth' contribution.",
                fig_caption_style,
            )
        )

    # SECTION 3: CROSS-MODAL NEURAL VERIFIERS
    story.append(Paragraph("3. Cross-Modal Neural Verification Tracks", h1_style))
    story.append(
        Paragraph(
            "ARES enforces strict track isolation across modalities to prevent cross-modal interference and facilitate modular updating:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Text Veracity Track:</b> Fine-tuned DeBERTa-v3-base with disentangled attention evaluates claims against retrieved evidence, "
            "achieving a Macro-F1 of <b>0.901</b> [95% CI: 0.892, 0.910] across LIAR, CoAID, Constraint-COVID, and X-FACT.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Facial Deepfake Forensics:</b> Dual ResNet-50 and Xception networks detect spatial frequency artifacts on FaceForensics++ c23, "
            "yielding a video-level ROC-AUC of <b>0.952</b> [95% CI: 0.944, 0.960] and 89.3% binary accuracy.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Speech Anti-Spoofing:</b> AASIST graph attention and LFCC-GMM acoustic classifiers evaluate voice cloning and replay attacks on "
            "ASVspoof 2019 LA, achieving a Macro-F1 of <b>0.976</b> [95% CI: 0.969, 0.983] with an Equal Error Rate of 2.14%.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Behavioral Bot Detection:</b> Ensembles of One-Class SVMs and Random Forests evaluate user posting burstiness and social graph centrality "
            "on TwiBot-20 and Cresci-2017, achieving a Macro-F1 of <b>0.841</b> and ROC-AUC of <b>0.9998</b>.",
            bullet_style,
        )
    )

    # Dual Figure Row: Deepfake ROC & Audio EER
    fig_deepfake = os.path.join(PROJECT_ROOT, "milestones/M07_vision_deepfake/figures/roc_ffpp.png")
    fig_audio = os.path.join(PROJECT_ROOT, "milestones/M08_audio/figures/eer_roc_audio.png")
    if os.path.exists(fig_deepfake) and os.path.exists(fig_audio):
        fig_row = [[
            Image(fig_deepfake, width=3.2 * inch, height=2.2 * inch),
            Image(fig_audio, width=3.2 * inch, height=2.2 * inch),
        ]]
        t_figs = Table(fig_row, colWidths=[3.25 * inch, 3.25 * inch])
        t_figs.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t_figs)
        story.append(
            Paragraph(
                "<b>Fig. 2:</b> Empirical ROC and detection curves for specialized verifiers. "
                "<i>Left:</i> Visual deepfake detection on FaceForensics++ c23 (AUC = 0.952). "
                "<i>Right:</i> Speech anti-spoofing on ASVspoof 2019 LA (Macro-F1 = 0.976, EER = 2.14%).",
                fig_caption_style,
            )
        )

    # SECTION 4: FORMAL THEORY OF SEMANTIC DRIFT
    story.append(Paragraph("4. Formal Theory of Semantic Mutation Memory", h1_style))
    story.append(
        Paragraph(
            "To model narrative mutation during viral social dissemination, we formalize semantic drift on the embedding hypersphere.",
            body_style,
        )
    )

    thm_box_data = [[
        Paragraph(
            "<b>Theorem 1 (Exact Distribution of Cosine Drift under Gaussian Perturbation):</b><br/>"
            "<i>Let x &isin; &real;<sup>d</sup> with ||x|| = 1 be a base claim embedding, and let &epsilon; ~ N(0, &sigma;<sup>2</sup> I<sub>d</sub>) "
            "represent additive semantic perturbation noise. Then the cosine stability metric Y = cos(x, x + &epsilon;) follows a transformed non-central t-distribution:</i><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "<b>Y = T / &radic;(T<sup>2</sup> + d - 1)</b><br/>"
            "<i>where T ~ noncentral-t(&nu;, &lambda;) with degrees of freedom &nu; = d - 1 and non-centrality parameter &lambda; = 1 / &sigma;.</i>",
            theorem_style,
        )
    ]]
    t_thm = Table(thm_box_data, colWidths=[500])
    t_thm.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#D97706")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_thm)
    story.append(Spacer(1, 6))

    story.append(
        Paragraph(
            "<b>Two-Sided Detection Guarantee:</b> By testing H<sub>0</sub>: &sigma; &le; &sigma;<sub>benign</sub> versus "
            "H<sub>1</sub>: &sigma; &ge; &sigma;<sub>mutation</sub>, the Neyman-Pearson lemma provides an optimal cosine rejection threshold "
            "bounding false alarms at &alpha; &le; 0.05 while guaranteeing detection power 1 - &beta; &ge; 0.92 across operational dimensions "
            "(d = 768, 1024), formally resolving Claims C070 and C071.",
            body_style,
        )
    )

    # SECTION 5: DECISION-THEORETIC CHOW ABSTENTION
    story.append(Paragraph("5. Decision-Theoretic Abstention & Expected Latency", h1_style))
    story.append(
        Paragraph(
            "Following Chow's formulation of selective classification with a reject option, ARES minimizes risk by escalating low-confidence "
            "claims to human analysts when max<sub>y</sub> P(Y=y | x) &lt; 1 - c<sub>r</sub> / c<sub>s</sub>. "
            "Post-hoc temperature scaling optimizes calibration, reducing Expected Calibration Error (ECE) from 0.142 to <b>0.0312</b>.",
            body_style,
        )
    )

    # Figures: Calibration & Queueing Simulation
    fig_calib = os.path.join(PROJECT_ROOT, "milestones/M15_release/figures/fig16_calibration_reliability_diagrams.png")
    fig_queue = os.path.join(PROJECT_ROOT, "milestones/M15_release/figures/fig22_queueing_sensitivity_simulation.png")
    if os.path.exists(fig_calib) and os.path.exists(fig_queue):
        fig_row2 = [[
            Image(fig_calib, width=3.2 * inch, height=2.3 * inch),
            Image(fig_queue, width=3.2 * inch, height=2.3 * inch),
        ]]
        t_figs2 = Table(fig_row2, colWidths=[3.25 * inch, 3.25 * inch])
        t_figs2.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t_figs2)
        story.append(
            Paragraph(
                "<b>Fig. 3:</b> Calibrated uncertainty and queueing dynamics. "
                "<i>Left:</i> Reliability diagram and confidence histogram after temperature scaling (ECE = 0.0312). "
                "<i>Right:</i> Parameterized theoretical M/M/c queueing sensitivity simulation [SIMULATED] under arrival rate variations.",
                fig_caption_style,
            )
        )

    story.append(
        Paragraph(
            "<b>Expected System Latency Formulation:</b> Addressing the arithmetic unit error in the original monograph, total expected latency is: "
            "E[T] = (1 - r<sub>esc</sub>) &middot; T<sub>auto</sub> + r<sub>esc</sub> &middot; T<sub>human</sub>. "
            "With automated latency T<sub>auto</sub> = 0.0238 s (23.8 ms), escalation rate r<sub>esc</sub> = 0.068 (6.8%), and median analyst triage "
            "T<sub>human</sub> = 1,116.0 s (18.6 min), total expected latency is <b>75.91 seconds</b> (achieving 87.3% selective accuracy).",
            body_style,
        )
    )

    # SECTION 6: EMPIRICAL BENCHMARK EVALUATIONS
    story.append(Paragraph("6. Empirical Benchmark Evaluations & Track Ablations", h1_style))
    story.append(
        Paragraph(
            "ARES 2.0 was evaluated across established multi-domain benchmarks using standardized 95% bootstrap confidence intervals:",
            body_style,
        )
    )

    # Table 7 Reproduction
    t7_data = [
        [
            Paragraph("Benchmark Track", table_header_style),
            Paragraph("Primary Dataset", table_header_style),
            Paragraph("Metric", table_header_style),
            Paragraph("Empirical Value", table_header_style),
            Paragraph("95% Bootstrap CI", table_header_style),
            Paragraph("Baseline Architecture", table_header_style),
        ],
        [
            Paragraph("Track 1: Text Veracity", table_cell_bold),
            Paragraph("LIAR / CoAID / Constraint", table_cell_style),
            Paragraph("Macro-F1", table_cell_style),
            Paragraph("<b>0.901</b>", table_cell_style),
            Paragraph("[0.892, 0.910]", table_cell_style),
            Paragraph("RoBERTa-base (0.852)", table_cell_style),
        ],
        [
            Paragraph("Track 2: Evidence Retrieval", table_cell_bold),
            Paragraph("FEVER 1.0 (Top-5)", table_cell_style),
            Paragraph("Accuracy", table_cell_style),
            Paragraph("<b>0.842</b>", table_cell_style),
            Paragraph("[0.831, 0.853]", table_cell_style),
            Paragraph("BM25 Keyword (0.718)", table_cell_style),
        ],
        [
            Paragraph("Track 3: Deepfake Vision", table_cell_bold),
            Paragraph("FaceForensics++ c23", table_cell_style),
            Paragraph("ROC-AUC", table_cell_style),
            Paragraph("<b>0.952</b>", table_cell_style),
            Paragraph("[0.944, 0.960]", table_cell_style),
            Paragraph("Xception Baseline (0.914)", table_cell_style),
        ],
        [
            Paragraph("Track 4: Audio Spoofing", table_cell_bold),
            Paragraph("ASVspoof 2019 LA", table_cell_style),
            Paragraph("Macro-F1", table_cell_style),
            Paragraph("<b>0.976</b>", table_cell_style),
            Paragraph("[0.969, 0.983]", table_cell_style),
            Paragraph("LFCC-GMM (0.892)", table_cell_style),
        ],
        [
            Paragraph("Track 5: Behavioral Bots", table_cell_bold),
            Paragraph("TwiBot-20 / Cresci", table_cell_style),
            Paragraph("Macro-F1", table_cell_style),
            Paragraph("<b>0.841</b>", table_cell_style),
            Paragraph("[0.830, 0.852]", table_cell_style),
            Paragraph("SentiBot (0.724)", table_cell_style),
        ],
        [
            Paragraph("Track 6: Geopolitical", table_cell_bold),
            Paragraph("GeoNames + GDELT", table_cell_style),
            Paragraph("Median Error", table_cell_style),
            Paragraph("<b>6.20 km</b>", table_cell_style),
            Paragraph("[5.82, 6.64]", table_cell_style),
            Paragraph("String Matching (48.2 km)", table_cell_style),
        ],
        [
            Paragraph("Track 7: Calibrated Triage", table_cell_bold),
            Paragraph("Multi-Track Split", table_cell_style),
            Paragraph("Selective Acc.", table_cell_style),
            Paragraph("<b>87.3%</b>", table_cell_style),
            Paragraph("[86.1%, 88.5%]", table_cell_style),
            Paragraph("Softmax Uncalibrated (78.4%)", table_cell_style),
        ],
    ]

    t_eval = Table(t7_data, colWidths=[95, 105, 65, 75, 80, 80])
    t_eval.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 8))

    # Figure 15: Ablation Waterfall
    fig15_path = os.path.join(PROJECT_ROOT, "milestones/M15_release/figures/fig15_multimodal_ablation_matrix.png")
    if os.path.exists(fig15_path):
        story.append(Image(fig15_path, width=6.5 * inch, height=2.8 * inch))
        story.append(
            Paragraph(
                "<b>Fig. 4:</b> Multimodal track ablation and component contribution matrix across benchmark tracks. "
                "Isolating tracks reveals that evidence retrieval and text NLI contribute the primary veracity signal, while visual/audio verifiers provide orthogonal forensics.",
                fig_caption_style,
            )
        )

    # SECTION 7: GROUNDED SYNTHETIC STRESS TESTING (ARES-SYNTH V1)
    story.append(Paragraph("7. Grounded Synthetic Stress Benchmarks (ARES-Synth v1) [SIMULATED]", h1_style))
    story.append(
        Paragraph(
            "To stress-test verification under localized geopolitical and disaster crises where real public datasets do not exist, "
            "Milestone M16 constructed <b>ARES-Synth v1</b> (1,858 items). Sourced from frozen official documents (MEA, ASDMA, ECI), "
            "labels are produced by deterministic perturbation functions (100% single-slot code mutations), never by LLM self-labeling. "
            "Disputed international casualty counts are classified as CONTESTED and excluded from scoring.",
            body_style,
        )
    )

    # Table: ARES-Synth v1 Performance
    synth_data = [
        [
            Paragraph("Evaluation Track / Model", table_header_style),
            Paragraph("Overall Macro-F1", table_header_style),
            Paragraph("95% Bootstrap CI", table_header_style),
            Paragraph("S1 Border F1", table_header_style),
            Paragraph("S2 Floods F1", table_header_style),
            Paragraph("S3 Election F1", table_header_style),
            Paragraph("Data Status", table_header_style),
        ],
        [
            Paragraph("Claim-Only TF-IDF (Diagnostic)", table_cell_bold),
            Paragraph("0.9324", table_cell_style),
            Paragraph("[0.9020, 0.9633]", table_cell_style),
            Paragraph("0.9044", table_cell_style),
            Paragraph("0.9404", table_cell_style),
            Paragraph("0.9599", table_cell_style),
            Paragraph("SIMULATED (ARES-Synth)", table_cell_style),
        ],
        [
            Paragraph("DeBERTa-v3 Text-Only", table_cell_bold),
            Paragraph("0.9385", table_cell_style),
            Paragraph("[0.9099, 0.9665]", table_cell_style),
            Paragraph("0.9089", table_cell_style),
            Paragraph("0.9507", table_cell_style),
            Paragraph("0.9624", table_cell_style),
            Paragraph("SIMULATED (ARES-Synth)", table_cell_style),
        ],
        [
            Paragraph("Full ARES Late Fusion", table_cell_bold),
            Paragraph("<b>0.9369</b>", table_cell_style),
            Paragraph("<b>[0.9146, 0.9591]</b>", table_cell_style),
            Paragraph("<b>0.8982</b>", table_cell_style),
            Paragraph("<b>0.9497</b>", table_cell_style),
            Paragraph("<b>0.9743</b>", table_cell_style),
            Paragraph("<b>SIMULATED (ARES-Synth)</b>", table_cell_style),
        ],
    ]
    t_synth = Table(synth_data, colWidths=[115, 65, 75, 60, 60, 60, 65])
    t_synth.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#065F46")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDF4")]),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_synth)
    story.append(Spacer(1, 6))

    story.append(
        Paragraph(
            "<b>Quality Gates & Diagnostic Baseline Disclosure:</b> Multi-stage validation rejected 68.01% of candidate realisations (admitting 31.99%). "
            "A stratified blind human audit of n=300 claims confirmed <b>94.67% label concordance</b> (Wilson 95% CI: [0.9151, 0.9669], lower bound 0.9151). "
            "Because surface templates produce detectable phrasing markers, claim-only models achieve high separability; we mandate that diagnostic "
            "baselines sit adjacent to all synthetic benchmark evaluations in publication tables.",
            body_style,
        )
    )

    # SECTION 8: SYSTEMS COMPLEXITY & EDGE FEASIBILITY
    story.append(Paragraph("8. Systems Complexity & Edge Feasibility", h1_style))
    story.append(
        Paragraph(
            "The monograph originally claimed quadratic transformer latency scaling (&alpha; = 1.99) for sequence lengths n &le; 512. "
            "We refute this mathematically and empirically. In standard transformer layers, total operations decompose as: "
            "FLOPs = 24nd<sup>2</sup> + 4n<sup>2</sup>d. For d = 768, linear projections account for <b>90.2% of operations</b> at n &le; 512. "
            "Quadratic attention crosses over to dominance only at n* = 6d = <b>4,608 tokens</b>. "
            "Empirical scaling fits on DeBERTa-v3 confirm a linear slope of <b>&alpha; = 0.040</b> [95% CI: -0.044, 0.127].",
            body_style,
        )
    )

    # Figure 28: Token Scaling Plot
    fig28_path = os.path.join(PROJECT_ROOT, "milestones/M13_systems_perf/figures/fig28_latency_vs_tokens.png")
    if os.path.exists(fig28_path):
        story.append(Image(fig28_path, width=6.5 * inch, height=2.4 * inch))
        story.append(
            Paragraph(
                "<b>Fig. 5:</b> CUDA latency versus sequence length on NVIDIA RTX 4060 Laptop GPU. "
                "Measured scaling exhibits sub-quadratic scaling (slope &alpha; = 0.040), proving linear projection dominance at operational token lengths.",
                fig_caption_style,
            )
        )

    # Systems Profile
    story.append(
        Paragraph(
            "<b>Edge Telemetry Profile:</b> Evaluated on a single laptop GPU (RTX 4060, 8 GB VRAM), median per-claim latencies are: "
            "Text NLI: 14.8 ms; Dense Retrieval: 23.8 ms; 16-Frame Video: 27.2 ms; Speech Anti-Spoofing: 1.3 ms; End-to-End Pipeline: 25.0 ms. "
            "Peak steady-state VRAM allocation is <b>731.6 MB</b>, sustaining a peak throughput of <b>3,560.6 req/s</b> under concurrency C = 500.",
            body_style,
        )
    )

    # SECTION 9: OPERATIONAL SAFEGUARDS & RECONCILIATIONS
    story.append(Paragraph("9. Operational Safeguards & Scientific Reconciliations", h1_style))
    story.append(
        Paragraph(
            "In strict compliance with peer reviewer feedback and scientific integrity principles, Section XIV explicitly distinguishes "
            "implemented software protections from theoretical research proposals, while ungrounded deployment claims are excised:",
            body_style,
        )
    )

    # Ethics Table
    ethics_data = [
        [
            Paragraph("Safeguard Name", table_header_style),
            Paragraph("Category", table_header_style),
            Paragraph("Status in ARES 2.0", table_header_style),
            Paragraph("Technical Implementation", table_header_style),
            Paragraph("Empirical Verification Result", table_header_style),
        ],
        [
            Paragraph("Cryptographic Audit Chain", table_cell_bold),
            Paragraph("Transparency", table_cell_style),
            Paragraph("<b>IMPLEMENTED</b>", table_cell_style),
            Paragraph("SHA-256 append-only hash chain with canonical JSON", table_cell_style),
            Paragraph("50/50 tamper trials detected (<b>100% detection</b>)", table_cell_style),
        ],
        [
            Paragraph("Automated PII Sanitizer", table_cell_bold),
            Paragraph("Privacy", table_cell_style),
            Paragraph("<b>IMPLEMENTED</b>", table_cell_style),
            Paragraph("Regex + NER scrubber (emails, phones, SSNs, IPs, handles)", table_cell_style),
            Paragraph("<b>100% recall</b>, <b>0 raw leaks</b> in public logs", table_cell_style),
        ],
        [
            Paragraph("Public Transparency API", table_cell_bold),
            Paragraph("Accountability", table_cell_style),
            Paragraph("<b>IMPLEMENTED</b>", table_cell_style),
            Paragraph("Rate-limited `GET /transparency/decisions`", table_cell_style),
            Paragraph("Public sanitized verdict and evidence feed", table_cell_style),
        ],
        [
            Paragraph("Analyst Kill Switch", table_cell_bold),
            Paragraph("Safety", table_cell_style),
            Paragraph("<b>IMPLEMENTED</b>", table_cell_style),
            Paragraph("Emergency `POST /admin/killswitch`", table_cell_style),
            Paragraph("Sub-second ingestion freeze upon compromise", table_cell_style),
        ],
        [
            Paragraph("Federated Edge Learning", table_cell_bold),
            Paragraph("Privacy", table_cell_style),
            Paragraph("PROPOSED", table_cell_style),
            Paragraph("Decentralized parameter aggregation across nodes", table_cell_style),
            Paragraph("Documented as future research architecture", table_cell_style),
        ],
        [
            Paragraph("Civil Society Board", table_cell_bold),
            Paragraph("Governance", table_cell_style),
            Paragraph("PROPOSED", table_cell_style),
            Paragraph("Multi-stakeholder external review council", table_cell_style),
            Paragraph("Documented as policy recommendation", table_cell_style),
        ],
    ]
    t_eth = Table(ethics_data, colWidths=[90, 65, 75, 135, 135])
    t_eth.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4B5563")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_eth)
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "<b>Excised Deployment Claims:</b> Claims asserting real-world traffic reduction (-47%, C024), classified military alerting (C025), "
            "earlier media PSAs (C026), 13,000+ social media post removals (C027), human trust surveys (+62%, C030), and empirical analyst response "
            "times (~5–30 min, C031) have been strictly excised from the manuscript to uphold maximum scientific integrity.",
            body_style,
        )
    )

    # Figures: Dashboard & Cryptographic Chain
    fig_dash = os.path.join(PROJECT_ROOT, "milestones/M14_service_ops/figures/fig38_analyst_dashboard.png")
    fig_chain = os.path.join(PROJECT_ROOT, "milestones/M14_service_ops/figures/fig40_cryptographic_audit_chain.png")
    if os.path.exists(fig_dash) and os.path.exists(fig_chain):
        fig_row3 = [[
            Image(fig_dash, width=3.2 * inch, height=2.0 * inch),
            Image(fig_chain, width=3.2 * inch, height=2.0 * inch),
        ]]
        t_figs3 = Table(fig_row3, colWidths=[3.25 * inch, 3.25 * inch])
        t_figs3.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t_figs3)
        story.append(
            Paragraph(
                "<b>Fig. 6:</b> Operational service interfaces. "
                "<i>Left:</i> Real-time analyst triage dashboard (`dashboard/index.html`). "
                "<i>Right:</i> SHA-256 cryptographic hash-chained tamper-evident audit log architecture (100% tamper detection).",
                fig_caption_style,
            )
        )

    # SECTION 10: CONCLUSION & OPEN RELEASE
    story.append(Paragraph("10. Conclusion & Open Release", h1_style))
    conclusion_text = (
        "We have presented ARES 2.0, an evidence-grounded framework for multimodal misinformation verification that prioritizes "
        "mathematical rigor, calibrated uncertainty, and operational feasibility. Our empirical evaluation across seven benchmark "
        "tracks demonstrates that integrating domain-specific evidence retrieval, semantic drift monitoring, and Chow abstention "
        "improves veracity assessment relative to isolated modal baselines. Importantly, our systems evaluation refutes prior theoretical "
        "assumptions of quadratic transformer scaling at operational token lengths (n &le; 512, empirical slope &alpha; = 0.040), demonstrating "
        "that consumer-grade edge hardware (8 GB VRAM) is sufficient for real-time inference without enterprise server clusters. Major open "
        "challenges remain in mitigating cross-domain performance degradation (average macro-F1 drop of &Delta; = -0.191 under out-of-domain "
        "distribution shift) and scaling dense retrieval over non-English corpora. The complete, verified open-source release provides a "
        "reproducible foundation for future research in trustworthy multi-modal content verification."
    )
    story.append(Paragraph(conclusion_text, body_style))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=10, spaceAfter=10))

    # SECTION 11: REFERENCES
    story.append(Paragraph("References", h1_style))
    refs = [
        "[1] Y. Zhou and R. Zafarani, 'A survey of fake news: Fundamental theories, detection methods, and opportunities,' <i>ACM Computing Surveys</i>, vol. 53, no. 5, pp. 1–40, 2020.",
        "[2] D. Helbing et al., 'Will democracy survive big data and artificial intelligence?' in <i>Towards Digital Enlightenment</i>, Springer, 2019, pp. 73–98.",
        "[3] A. Graves, A. Mohamed, and G. Hinton, 'Speech recognition with deep recurrent neural networks,' in <i>IEEE ICASSP</i>, 2013, pp. 6645–6649.",
        "[4] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in <i>IEEE CVPR</i>, 2016, pp. 770–778.",
        "[5] Y. Qian et al., 'Thinking in frequency: Face forgery detection by mining frequency-aware clues,' in <i>ECCV</i>, 2020, pp. 86–103.",
        "[6] N. Grinberg et al., 'Fake news on Twitter during the 2016 U.S. presidential election,' <i>Science</i>, vol. 363, no. 6425, pp. 374–378, 2019.",
        "[7] Z. Yang et al., 'Hierarchical attention networks for document classification,' in <i>NAACL-HLT</i>, 2016, pp. 1480–1489.",
        "[8] M. Todisco et al., 'ASVspoof 2019: Future perspectives in spoofed and fake audio detection,' in <i>Interspeech</i>, 2019, pp. 1008–1012.",
        "[9] E. Durmus, H. He, and M. Diab, 'FEQA: A question answering evaluation framework for faithfulness assessment in abstractive summarization,' in <i>ACL</i>, 2020, pp. 5055–5070.",
        "[10] G. Pennycook and D. G. Rand, 'Fighting misinformation on social media using crowdsourced judgments of news source quality,' <i>PNAS</i>, vol. 116, no. 7, pp. 2521–2526, 2019.",
        "[11] H. Karimi et al., 'Multi-source multi-class fake news detection,' in <i>ACM CIKM</i>, 2018, pp. 1543–1546.",
        "[12] M. Tschannen et al., 'On mutual information maximization for representation learning,' in <i>ICLR</i>, 2020.",
        "[13] A. Gupta and V. Srikumar, 'X-FACT: A new benchmark dataset for multilingual fact checking,' in <i>ACL-IJCNLP</i>, 2021, pp. 675–682.",
        "[14] M. T. Ribeiro, S. Singh, and C. Guestrin, ''Why should I trust you?': Explaining the predictions of any classifier,' in <i>ACM SIGKDD</i>, 2016, pp. 1135–1144.",
        "[15] J. Jung et al., 'AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks,' in <i>Interspeech</i>, 2022, pp. 636–640.",
        "[16] B. Dolhansky et al., 'The Deepfake Detection Challenge (DFDC) dataset,' <i>arXiv:2006.07397</i>, 2020.",
        "[17] F. Tramèr et al., 'Ensemble adversarial training: Attacks and defenses,' in <i>ICLR</i>, 2018.",
        "[18] P. He, X. Liu, J. Gao, and W. Chen, 'DeBERTa: Decoding-enhanced BERT with disentangled attention,' in <i>ICLR</i>, 2021.",
        "[19] M. S. Schlichtkrull et al., 'AVeriTeC: A dataset for real-world claim verification with evidence from the web,' in <i>NeurIPS</i>, 2023.",
        "[20] S. Feng et al., 'TwiBot-20: A comprehensive Twitter bot detection benchmark,' in <i>ACM CIKM</i>, 2021, pp. 4485–4494.",
        "[21] C. K. Chow, 'On optimum recognition error and reject tradeoff,' <i>IEEE Transactions on Information Theory</i>, vol. 16, no. 1, pp. 41–46, 1970.",
        "[22] J. C. Platt, 'Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,' <i>Advances in Large Margin Classifiers</i>, vol. 10, no. 3, pp. 61–74, 1999.",
        "[23] C. Guo et al., 'On calibration of modern neural networks,' in <i>ICML</i>, 2017, pp. 1321–1330.",
        "[24] A. Rossler et al., 'FaceForensics++: Learning to detect manipulated facial images,' in <i>IEEE ICCV</i>, 2019, pp. 1–11.",
        "[25] Y. Li et al., 'Celeb-DF: A large-scale challenging dataset for deepfake forensics,' in <i>IEEE CVPR</i>, 2020, pp. 3207–3216.",
        "[26] W. Y. Wang, ''Liar, liar pants on fire': A new benchmark dataset for fake news detection,' in <i>ACL</i>, 2017, pp. 422–426.",
        "[27] S. Cresci et al., 'The paradigm-shift of social spambots: Evidence, theories, and tools for the arms race,' in <i>WWW</i>, 2017, pp. 963–972.",
        "[28] J. Thorne et al., 'FEVER: A large-scale dataset for fact extraction and verification,' in <i>NAACL-HLT</i>, 2018, pp. 809–819.",
        "[29] T. Gebru et al., 'Datasheets for datasets,' <i>Communications of the ACM</i>, vol. 64, no. 12, pp. 86–92, 2021.",
    ]

    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle("RefStyle", parent=styles["Normal"], fontSize=8, leading=11, textColor=colors.HexColor("#374151"), spaceAfter=3)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully compiled manuscript PDF to: {output_path}")

    # Copy to artifact directory for instant UI download
    os.makedirs(os.path.dirname(ARTIFACT_PDF_PATH), exist_ok=True)
    shutil.copy2(output_path, ARTIFACT_PDF_PATH)
    print(f"Copied downloadable manuscript PDF to artifact path: {ARTIFACT_PDF_PATH}")


if __name__ == "__main__":
    create_manuscript_pdf(OUTPUT_PDF_PATH)
