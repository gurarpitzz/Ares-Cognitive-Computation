# M07 Worklog — Visual Deepfake Detection

## Phase 0: Orientation & Dependency Review
- **Date & Time:** 2026-10-13T19:55:00+05:30
- **Hardware Profile:** HP OMEN Gaming Laptop (Intel Core i7-13620H 16 threads, NVIDIA GeForce RTX 4060 Laptop GPU 8 GB VRAM, 16 GB DDR5, Windows 11 Home 24H2 / WSL2 Ubuntu).
- **Target Ledger Claims:**
  - `C013`: ARES deepfake (FF++): TPR/FPR/F1 reported 95% / 2.5% / 0.97.
  - `C015`: CNN baseline: TPR/FPR/F1 reported 89% / 8.0% / 0.92.
- **Audit Findings to Address:**
  - `Issue I-14`: Table XI reported $F_1$ scores are mathematically impossible under balanced classes ($F_1 = 0.962$ vs reported $0.97$ for ARES; $F_1 = 0.904$ vs reported $0.92$ for CNN baseline). We establish the closed-form theorem relating $F_1$, $\text{TPR}$, $\text{FPR}$, and class ratio $r = N_{\text{real}}/N_{\text{fake}}$, proving that the reported values require an undisclosed class imbalance ($r \approx 0.25$ for FF++ test split) or were rounded up.
- **Operational Design & Red-Flag Checklist:**
  - `RF1 (Identity Disjointness)`: Train, validation, and test splits must contain zero identity/video overlap.
  - `RF2 (Face-Detection Accounting)`: Videos where face extraction is challenging must be accounted for and scored with uncertainty ($p=0.5$), never silently omitted.
  - `RF3 (Threshold Selection)`: Operating thresholds (Youden $J$, fixed $\text{FPR} = 5\%$) must be selected strictly on validation data.
  - `RF4 (Class Prior Disclosure)`: $F_1$ must always be reported alongside the exact real:fake ratio ($r$); $\text{ROC-AUC}$ remains primary.
  - `RF5 (Cross-Dataset Generalization)`: Celeb-DF evaluation must be strictly zero-shot with zero Celeb-DF samples in training or validation.
  - `RF6 (Plausibility vs Literature)`: In-domain FF++ AUC is expected to be high ($\ge 0.90$), while cross-dataset Celeb-DF AUC drops substantially ($0.65 - 0.78$), verifying generalization gap.
