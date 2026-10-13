# ARES 2.0 Master Reproducibility Guide (Claim C099)

This guide provides the complete, deterministic instructions for reproducing every benchmark, statistical test, latency measurement, and table/figure published in the ARES 2.0 evaluation.

---

## 1. Hardware & System Prerequisites

All benchmarks and latency profiles are measured and validated on the following hardware platform:
- **Device:** HP OMEN 16-wd0xxx Laptop
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6 VRAM, TGP 140W)
- **CPU:** 13th Gen Intel Core i7-13620H (10 cores, 16 threads, up to 4.9 GHz)
- **System Memory:** 16 GB DDR5 5200 MHz RAM
- **Operating System:** Windows 11 Home 24H2 with WSL2 (Ubuntu 26.04 LTS)
- **Software Stack:** Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 5.6.2

---

## 2. Environment Initialization

```bash
# Clone the repository
git clone https://github.com/gurarpitzz/Ares-Cognitive-Computation.git
cd Ares-Cognitive-Computation
git checkout v2.0.0

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .venv\Scripts\Activate.ps1

# Install exact pinned dependencies
pip install -r requirements.txt
pip install -e ares_v2/
```

---

## 3. Milestone-by-Milestone Reproduction Commands

Every milestone is fully automated via dedicated evaluation runners in `ares_v2/scripts/`:

| Milestone | Target Domain / Component | Execution Command | Approx. Runtime | Target Ledger Claims |
| :--- | :--- | :--- | :---: | :--- |
| **M01** | Mathematical Theory & Mutation Theorems | `python ares_v2/scripts/run_m01_evaluation.py` | 10 sec | `C070`–`C075`, `C077`, `C078` |
| **M02** | Data Curation, Splits & Leakage Audit | `python ares_v2/scripts/run_m02_evaluation.py` | 30 sec | `C032`, `C088`, `C089` |
| **M03** | Layer-1 Text Verifiers (DeBERTa-v3) | `python ares_v2/scripts/run_m03_evaluation.py` | 45 sec | `C038`, `C039`, `C050`, `C064`, `C065` |
| **M04** | Sparse/Dense Evidence Engine ($R \cdot S \cdot N \cdot W$) | `python ares_v2/scripts/run_m04_evaluation.py` | 25 sec | `C018`, `C040`, `C067` |
| **M05** | Decision Layer, Calibration & Chow Abstention | `python ares_v2/scripts/run_m05_evaluation.py` | 15 sec | `C010`, `C061`–`C063`, `C072` |
| **M06** | Claim Mutation & Semantic Drift ($\Delta_t$) | `python ares_v2/scripts/run_m06_evaluation.py` | 20 sec | `C011`, `C012`, `C085`, `C086` |
| **M07** | Visual Deepfake Forensics (ResNet-50) | `python ares_v2/scripts/run_m07_evaluation.py` | 35 sec | `C013`, `C015` |
| **M08** | Speech Spoofing & Audio Verification | `python ares_v2/scripts/run_m08_evaluation.py` | 15 sec | `C014`, `C015`, `C088` |
| **M09** | Behavioral Stylometry & Bot Detection | `python ares_v2/scripts/run_m09_evaluation.py` | 20 sec | `C080`, `C081` |
| **M10** | Geopolitical Context, GDELT & Risk $\rho$ | `python ares_v2/scripts/run_m10_evaluation.py` | 25 sec | `C007`–`C009`, `C083` |
| **M11** | Multimodal Track Fusion & Ablation | `python ares_v2/scripts/run_m11_evaluation.py` | 50 sec | `C001`, `C043`, `C046`–`C049`, `C051`–`C058` |
| **M12** | Adversarial Robustness & ARES-Synth v1 | `python ares_v2/scripts/run_m12_evaluation.py` | 40 sec | `C003`, `C019`–`C023`, `C037`, `C044`, `C045` |
| **M13** | Systems Latency, Throughput & Scaling | `python ares_v2/scripts/run_m13_evaluation.py` | 55 sec | `C002`, `C004`–`C006`, `C016`, `C017`, `C028`–`C036` |
| **M14** | Operational Service, Dashboard & Audit Chain | `python ares_v2/scripts/run_m14_evaluation.py` | 5 sec | `C031`, `C040`, `C082`, `C083`, `C084`, `C093` |
| **M15** | Release Synthesis, Meta-Audit & References | `python ares_v2/scripts/run_m15_evaluation.py` | 10 sec | `C024`–`C027`, `C030`, `C076`, `C079`, `C090`, `C092`, `C099` |

---

## 4. Master Unit Test Suite

Run the full automated pytest suite verifying all verifiers, fusion heads, audit chains, and API endpoints:
```bash
pytest ares_v2/tests -q
```
**Expected Outcome:** 110 passed, 0 failed, 0 warnings.

---

## 5. Audit Log & Service Verification

To independently verify the append-only cryptographic audit chain:
```bash
python ares_v2/scripts/verify_auditlog.py milestones/M14_service_ops/data/audit_log.jsonl
```
**Expected Outcome:**
```
[PASS] Audit log verified successfully: 40 blocks, 0 errors.
```

To run the clean-room smoke reproduction audit:
```bash
python ares_v2/scripts/m15/cleanroom_smoke.py
```
**Expected Outcome:** `cleanroom_verdict: PASS` in `milestones/M15_release/results/cleanroom_report.json`.
