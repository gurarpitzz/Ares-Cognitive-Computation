# M00 Compute Budget & GPU-Hour Re-Estimation

**Hardware Target:** HP OMEN 16-wd0xxx  
**GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB GDDR6, Ada Lovelace)  
**CPU:** 13th Gen Intel Core i7-13620H (10 Cores, 16 Threads)  
**RAM:** 16 GB DDR5 (12 GB allocated to WSL2)  
**Date:** October 13, 2026  

---

## 1. Initial Plan Estimates vs Workstation Profile

In `plan/03_IMPLEMENTATION_PLAN.md` §5.5, estimated total GPU wall-clock was projected at **350–450 GPU-hours** across all milestones. Based on VRAM fit profiling and hardware constraints on the RTX 4060 Laptop (8 GB), we refine the execution plan to fit within thermal limits and storage boundaries (<= 150 GB on D:).

| Milestone | Task | Model Family | VRAM Tactic | Estimated GPU-h | Parallel CPU Work |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M00** | Foundation & Smoke Tests | Matmul / Transformer-base | bf16 micro-batch 8–16 | ~1–2 | Repo scaffold, ledger tooling, lockfiles |
| **M01** | Theory & Exact Distributions | Analytical / Monte Carlo | CPU (NumPy/SciPy) | 0 (CPU only) | Exact distributions, LLR derivation, proofs |
| **M02** | Data Acquisition & Curation | Ingestion & Feature extraction | Preprocessing / Face crop | ~10–12 | Download pipelines, split manifests, hashing |
| **M03** | Layer-1 Text Verifiers | DeBERTa-v3-base (K=10 seeds) | bf16, micro-batch 8, accum 4 (eff 32) | ~80–100 | Metric computation, Paired bootstrap |
| **M04** | Evidence Engine & Logic | BM25 + MiniLM + BART-MNLI | CPU BM25 + GPU NLI inference | ~15–20 | Pyserini indexing, logic rules |
| **M16** | ARES-Synth Benchmarks | Grounded Synthetic Generators | Batch NLI verification | ~6–8 | Perturbation generation, Wikipedia/ReliefWeb |
| **M05** | Decision & Calibration | Logistic LLR / Platt / Chow | CPU/GPU lightweight | ~4–5 | Risk-coverage curves, cost grids |
| **M06** | Drift & Mutation Tracking | Exact noncentral-t / Delta_t | Embedding extraction | ~3–4 | Hypothesis testing, ROC curves |
| **M07** | Vision Deepfake Detection | ResNet-50 / EfficientNet-B4 | Face crops, bf16 | ~50–60 | Video frame extraction (CPU/ffmpeg) |
| **M08** | Audio Forensics & Spoof | AASIST / Log-mel CNN / ECAPA | Audio spectrograms, bf16 | ~40–45 | Audio feature caching |
| **M09** | Behavioral & Bot Detection | Stylometry + OCSVM / UPFD GCN | GPU GCN / CPU OCSVM | ~4–5 | Graph construction |
| **M10** | Geopolitical Context Engine | GDELT / Wikidata / Factbook | CPU snapshotting / NER | ~3–4 | Knowledge graph links |
| **M11** | Track Fusion & Ablation | Track Fusion Heads / Shapley | Cached feature inputs (fast) | ~35–40 | Exact Shapley attribution |
| **M12** | Robustness & Stress Testing | Adversarial attacks / Shift | Evaluation on frozen tracks | ~45–55 | LOSO cross-domain evaluation |
| **M13** | Systems Performance & Saturation| Latency / Throughput / Scaling | Thermal protocol on RTX 4060 | ~8–10 | Saturation benchmarks |
| **M14** | Production Service & Ops | FastAPI / Audit chaining | Local server & benchmark | ~2–3 | PII sanitization benchmarks |
| **M15** | Release & Paper Reconciliation | Package verification & artifact build | Clean-room verification | ~2–3 | Vector figure rendering, LaTeX |
| **TOTAL** | | | | **~300–375 GPU-h** | **Overnight / Unattended Queue** |

---

## 2. Resource Management Strategies

1. **VRAM Fit Assurance:**
   - 8 GB GDDR6 capacity is sufficient for all base encoder fine-tuning (DeBERTa-v3-base, RoBERTa-base, mDeBERTa) using `bfloat16` and gradient accumulation ($8 \times 4 = 32$ effective batch).
   - Any LLM evaluation uses 4-bit quantisation (bitsandbytes NF4 or GGUF Q4_K_M).
2. **Thermal Stability:**
   - OMEN Gaming Hub set to Performance.
   - Background telemetry logs GPU temperatures every 30s. If temperatures hit $\ge 87^\circ\text{C}$ with clock drop $> 20\%$ for $> 5$ minutes, jobs are flagged and throttled runs excluded from timing baselines.
3. **Storage Cap ($\le 150\text{ GB}$ on D:):**
   - Raw video archives (FaceForensics++) are deleted immediately following face crop extraction and manifest verification.
   - Large intermediate checkpoints pruned to keep only the seed-median model and final best epoch.
