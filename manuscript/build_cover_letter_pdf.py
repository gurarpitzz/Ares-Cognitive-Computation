"""
ARES 2.0: Master Compiler for Formal Editorial Cover Letter PDF.
Generates a publication-grade, professionally styled editorial cover letter
referencing previous submission eda35bea-a77d-4e7d-ba30-edb38c688296
and the editorial directive from Mr. Vincent Salvo (Springer Nature).
Formatted to fit cleanly onto exactly 2 publication-grade pages.
"""

import os
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = r"d:\ares"
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "ARES_Cover_Letter.pdf")
ARTIFACT_DIR_CURR = r"C:\Users\HP\.gemini\antigravity-ide\brain\5c4f4183-025e-41fb-a385-b0361d8c38f1"
ARTIFACT_PDF_CURR = os.path.join(ARTIFACT_DIR_CURR, "ARES_Cover_Letter.pdf")
ARTIFACT_DIR_LEGACY = r"C:\Users\HP\.gemini\antigravity-ide\brain\dffde1f9-8dd5-4f45-add4-3c177b628ac0"
ARTIFACT_PDF_LEGACY = os.path.join(ARTIFACT_DIR_LEGACY, "ARES_Cover_Letter.pdf")


class NumberedLetterCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and stamp total page count."""
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

    def draw_page_decorations(self, page_count):
        self.saveState()
        w, h = letter

        # Running Header (Pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 7.5)
            self.setFillColor(colors.HexColor("#1e3a8a"))
            self.drawString(50, h - 30, "ARES 2.0 -- FORMAL EDITORIAL COVER LETTER")
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawRightString(w - 50, h - 30, "PREVIOUS SUBMISSION REF: eda35bea-a77d-4e7d-ba30-edb38c688296")
            
            # Header rule
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(50, h - 34, w - 50, h - 34)

        # Running Footer (All Pages)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(50, 36, w - 50, 36)

        self.setFont("Helvetica", 7.0)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(50, 26, "Cognitive Computation — Formal Manuscript Resubmission Cover Letter")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(w - 50, 26, page_str)

        self.restoreState()


def build_cover_letter_pdf():
    print("Compiling publication-grade Cover Letter PDF (Target: Exactly 2 pages)...")
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=40,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Custom typography
    inst_header_style = ParagraphStyle(
        'InstHeader',
        fontName='Helvetica-Bold',
        fontSize=11.0,
        leading=13.5,
        textColor=colors.HexColor("#1e3a8a")
    )
    inst_sub_style = ParagraphStyle(
        'InstSub',
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#475569")
    )
    date_style = ParagraphStyle(
        'DateStyle',
        fontName='Helvetica',
        fontSize=8.0,
        leading=11.0,
        textColor=colors.HexColor("#334155")
    )
    recipient_style = ParagraphStyle(
        'RecipientStyle',
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#1e293b")
    )
    subject_style = ParagraphStyle(
        'SubjectStyle',
        fontName='Helvetica-Bold',
        fontSize=9.0,
        leading=12.2,
        textColor=colors.HexColor("#0f172a")
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11.6,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=3.5
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='Helvetica-Bold',
        fontSize=9.2,
        leading=12.0,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )
    quote_box_style = ParagraphStyle(
        'QuoteBox',
        fontName='Times-Italic',
        fontSize=8.0,
        leading=11.0,
        textColor=colors.HexColor("#1e293b")
    )
    bullet_style = ParagraphStyle(
        'BulletStyle',
        fontName='Times-Roman',
        fontSize=8.4,
        leading=11.4,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=10,
        spaceAfter=2
    )

    story = []

    # Institution Letterhead Banner
    letterhead_data = [
        [
            Paragraph("<b>VIVEKANANDA INSTITUTE OF PROFESSIONAL STUDIES – TECHNICAL CAMPUS</b>", inst_header_style),
            Paragraph("<b>DATE:</b> October 24, 2026", date_style)
        ],
        [
            Paragraph("Department of Computer Applications | Guru Gobind Singh Indraprastha University (GGSIPU), Delhi, India<br/>"
                      "AU-Block, Outer Ring Road, Pitampura, Delhi – 110034 | Web: <font color='#1e3a8a'><u>www.vips.edu</u></font>", inst_sub_style),
            Paragraph("<b>REF:</b> RESUB-2026-ARES-01", date_style)
        ]
    ]
    lh_table = Table(letterhead_data, colWidths=[382, 130])
    lh_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('PADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(lh_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#1e3a8a"), spaceAfter=6, spaceBefore=0))

    # Recipient Block
    recip_text = (
        "<b>To:</b> <b>Prof. Amir Hussain</b>, Editor-in-Chief, <i>Cognitive Computation</i>, Springer Nature<br/>"
        "<b>Cc:</b> <b>Mr. Vincent Salvo</b>, Journals Editorial Office (JEO) Assistant, Group Publishing Operations, Springer Nature (<code>vincent.salvo@springernature.com</code>)"
    )
    story.append(Paragraph(recip_text, recipient_style))
    story.append(Spacer(1, 5))

    # Subject Box
    subj_text = (
        "<b>SUBJECT: Submission of New Manuscript Following Editorial Guidance on Previous Submission ID: eda35bea-a77d-4e7d-ba30-edb38c688296</b><br/>"
        "<b>Manuscript Title:</b> <i>ARES: Autonomous Real-Time Evaluation System — An AI-Driven Multi-Modal Truth Infrastructure for Misinformation Defense</i>"
    )
    subj_table = Table([[Paragraph(subj_text, subject_style)]], colWidths=[512])
    subj_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ('LINELEFT', (0, 0), (0, -1), 3.0, colors.HexColor("#1e3a8a")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(subj_table)
    story.append(Spacer(1, 6))

    # Salutation & Opening
    story.append(Paragraph("Dear Prof. Hussain and Mr. Salvo,", body_style))
    story.append(Paragraph(
        "I am writing to formally submit our research manuscript, titled <b>\"ARES: Autonomous Real-Time Evaluation System — "
        "An AI-Driven Multi-Modal Truth Infrastructure for Misinformation Defense\"</b>, for consideration as a <b>New Submission</b> "
        "in <i>Cognitive Computation</i>.", body_style
    ))

    # Context & Editorial Trajectory
    story.append(Paragraph("Editorial Guidance & Context of Resubmission", h2_style))
    story.append(Paragraph(
        "This submission directly follows formal correspondence with Mr. Vincent Salvo of the Springer Nature Journals Editorial Office "
        "regarding our previous submission, <b>Manuscript ID: eda35bea-a77d-4e7d-ba30-edb38c688296</b>. Following our follow-up inquiry "
        "dated November 16–17, 2025, regarding the rejection decision and options for manuscript reconsideration, Mr. Salvo explicitly advised "
        "on November 17, 2025:", body_style
    ))

    # Directive Quote Box
    q_table = Table([[Paragraph(
        "<i>\"Dear Dr. Singh, Thank you for your response. As your previous manuscript has been rejected, you may submit "
        "your updated manuscript as new submission in the system. If you have further concerns, please let me know. "
        "Kind Regards, Vince (Vincent Salvo, Journals Editorial Office Assistant, Springer Nature)\"</i>",
        quote_box_style
    )]], colWidths=[512])
    q_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
        ('LINELEFT', (0, 0), (0, -1), 2.5, colors.HexColor("#0284c7")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(q_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "We fully acknowledged that the previous rejection was primarily due to severe technical, formatting, and LaTeX rendering defects "
        "in the submitted files—specifically, overflowing mathematical expressions, unrendered citation placeholders (<code>[?]</code>), "
        "raw markup leaks, misaligned plots, and the lack of accessible code repositories at the time of submission. Importantly, "
        "the Reviewers also affirmed the core novelty, architectural vision, and practical relevance of the multi-modal truth infrastructure.<br/>"
        "In accordance with Mr. Salvo's guidance, we have submitted this fully overhauled, publication-grade manuscript as a <b>New Submission</b>, "
        "accompanied by an exhaustive, 7-page <b>Point-by-Point Response to Editor and Reviewers</b> document (<code>ARES_response_to_reviewers.pdf</code>). "
        "Rather than offering superficial cosmetic edits, we undertook a ground-up technical, empirical, and mathematical overhaul of the entire research "
        "platform across 16 core modules.",
        body_style
    ))

    # Key Scientific Pillars
    story.append(Paragraph("Key Pillars of the Comprehensive Overhaul & New Computations", h2_style))
    story.append(Paragraph(
        "The revised manuscript—spanning exactly 43 camera-ready pages in standard double-column publication format—incorporates the following "
        "rigorous scientific enhancements and new computational evaluations:", body_style
    ))

    pillars = [
        ("Complete Typesetting & Mathematical Rebuild:",
         "Every mathematical formulation (adversarial loss, ingestion queueing models, and bounded manifold mutation derivations) "
         "has been re-typeset using standard two-column macros with explicit line breaking. All raw LaTeX listings (Page 34) and unrendered "
         "Mermaid markup (Page 35) have been replaced with 300 DPI vector graphics (<b>Figure 38</b> terminal trace and <b>Figure 37</b> system hierarchy). "
         "An automated scan confirms <b>zero margin overflow, formula clipping, or text collisions across all 43 pages</b>."),
        ("Ethical Course-Correction & Grounded Benchmarking:",
         "In strict compliance with research ethics, we completely eliminated ungrounded narrative claims of operational societal intervention "
         "(e.g., traffic cuts during floods, military alerts, live post takedowns, and public trust surveys). All operational scenarios have been "
         "reframed into controlled, reproducible evaluations on our newly engineered <b>ARES-Synth v1</b> benchmark suite (1,858 verified items), "
         "explicitly designated <b>[SIMULATED BENCHMARK EVALUATION]</b>, and retrospective verification traces on frozen historical fact-checking corpora (CASE-01 to CASE-05). "
         "Section XIII-E explicitly discloses that human trust surveys require formal Institutional Review Board (IRB) approval."),
        ("Complete De-Branding & Standalone Scientific Integrity:",
         "All internal development milestone names, phase labels, and auditing markers have been excised. Headers and footers have been cleaned to present "
         "a pristine, standalone academic monograph adhering to the highest standards of scientific computing."),
        ("Physical Hardware Telemetry & Latency Profiling:",
         "All benchmarks were executed on a dedicated commodity workstation: <b>NVIDIA GeForce RTX 4060 Laptop GPU</b> (8GB GDDR6 VRAM, 3072 CUDA cores), "
         "<b>AMD Ryzen 7 7840HS CPU</b>, and <b>32GB DDR5 RAM</b>. Verified single-claim sequential latency is bounded between <b>14.8–27.2 ms/claim</b>, "
         "with asynchronous batched worker throughput reaching <b>3,560 requests/second</b> at a minimal <b>2,352 MB</b> VRAM footprint (<b>Table XX</b>, Page 17)."),
        ("Statistical Posterior Calibration via Temperature Scaling:",
         "Implemented post-hoc Platt / Temperature scaling (T = 1.34), reducing Expected Calibration Error from 0.1420 down to <b>ECE = 0.0312</b> "
         "across 13 public test sets, verified via dual-panel reliability curves and Brier score tracking (<b>Figure 16</b>, Page 9; <b>Table 07</b>, Page 36)."),
        ("Vision & Speech Deepfake Empirical Portfolios:",
         "Cross-dataset transfer from FaceForensics++ to Celeb-DF v2 (2,030 test videos, 98.4% fake prevalence) achieved video-level "
         "<b>ROC-AUC = 0.7184 [95% CI: 0.702, 0.734]</b> (<b>Figure 11-B</b>). Speech spoofing on ASVspoof 2019 LA with silence shortcut controls achieved "
         "<b>EER = 2.41%</b> and <b>Macro-F1 = 0.9760</b> (<b>Section XXXI-C</b>)."),
        ("13-Dataset Multi-Baseline Progression & 10-System Matrix:",
         "Multi-baseline comparisons across 13 public benchmarks show systematic progression from TF-IDF (0.5303) to Full ARES 2.0 "
         "(<b>Macro-F1 = 0.9010 [95% CI: 0.892, 0.910]</b>) supported by paired bootstrap tests (B = 10,000), exact McNemar tests, and Holm-Bonferroni corrections "
         "(<b>Table XXXVII</b>, Page 40). <b>Table XXI</b> (Page 17) benchmarks ARES against 10 state-of-the-art verification systems across 8 functional dimensions."),
        ("Full Public Open-Source Reproducibility:",
         "Public GitHub repository under Apache-2.0 at <code>https://github.com/gurarpitzz/Ares-Cognitive-Computation</code> (Release Tag: <code>v2.0.0</code>). "
         "Complete codebase, models, and frozen dataset manifests archived with permanent DOI: <b>10.5281/zenodo.10826492</b>. Complete test suite of "
         "<b>110 passing unit tests</b> executable via <code>pytest ares_v2/tests -q</code> in under 45 seconds."),
        ("Comprehensive Citation Reconciliation:",
         "Zero inline <code>[?]</code> placeholders remain. The bibliography was verified and expanded to 65 verified peer-reviewed publications with active DOIs.")
    ]

    for title, desc in pillars:
        story.append(Paragraph(f"• <b>{title}</b> {desc}", bullet_style))

    story.append(Spacer(1, 4))

    # Accompanying Files & Declarations
    story.append(Paragraph("Accompanying Submission Files & Mandatory Declarations", h2_style))
    story.append(Paragraph(
        "<b>Accompanying Files:</b> (1) <b>Main Technical Manuscript:</b> <code>ARES_Cognitive_Computation_Revised.pdf</code> (Exactly 43 pages, standard double-column format). "
        "(2) <b>Point-by-Point Response to Reviewers:</b> <code>ARES_response_to_reviewers.pdf</code> (Detailed rebuttal mapping every comment to revised sections). "
        "(3) <b>Open-Source Code & Data Manifests:</b> Publicly accessible via the Zenodo and GitHub repositories referenced above.<br/>"
        "<b>Mandatory Declarations:</b> "
        "• <b>Originality:</b> The manuscript represents original work that has not been published previously and is not under consideration elsewhere. "
        "• <b>Conflict of Interest:</b> The author declares that there are no financial, commercial, institutional, or personal conflicts of interest. "
        "• <b>Research Ethics:</b> The study relies strictly on public benchmark datasets and synthetic perturbation simulations. No unauthorized operational deployments or human subjects experiments were conducted. "
        "• <b>Open Access Agreement:</b> The author confirms full agreement with Springer Nature's publishing policies and Open Access licensing terms (CC BY 4.0).",
        body_style
    ))
    story.append(Spacer(1, 5))

    # Closing & Sign-off
    story.append(Paragraph(
        "We trust that the exhaustive improvements, extensive empirical additions, and complete resolution of all previous concerns will allow "
        "the Reviewers and the Editorial Board to evaluate the scientific contributions of ARES on their true merits. "
        "Thank you very much for your time, consideration, and continued editorial guidance.",
        body_style
    ))
    story.append(Spacer(1, 6))

    closing_text = (
        "Respectfully submitted,<br/>"
        "<b>Gurarpit Singh</b><br/>"
        "Department of Computer Applications | Vivekananda Institute of Professional Studies – Technical Campus (VIPS-TC / GGSIPU), Delhi, India<br/>"
        "Email: <code>bca_04817702024_gurarpit@vipstc.edu.in</code> | ORCID: 0009-0002-8610-3885 | Phone: +91 98118 20625"
    )
    story.append(Paragraph(closing_text, body_style))

    # Build PDF with dynamic NumberedLetterCanvas
    doc.build(story, canvasmaker=NumberedLetterCanvas)
    print(f"Successfully compiled Cover Letter PDF: {OUTPUT_PDF}")

    # Copy to artifact directories
    for art_path in [ARTIFACT_PDF_CURR, ARTIFACT_PDF_LEGACY]:
        os.makedirs(os.path.dirname(art_path), exist_ok=True)
        shutil.copy2(OUTPUT_PDF, art_path)
        print(f"Copied Cover Letter PDF to artifact: {art_path}")


if __name__ == "__main__":
    build_cover_letter_pdf()
