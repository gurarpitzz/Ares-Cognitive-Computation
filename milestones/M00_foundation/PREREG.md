# M00 Pre-Registration — Foundation & Integrity Infrastructure

**Milestone:** M00 — Foundation, Integrity Infrastructure & Access Requests  
**Date:** 2026-10-13  
**Status:** PRE-REGISTERED  
**Target Ledger Claim:** C097 (Bootstrap CI method) + all ledger tooling  
**Hardware Target:** HP OMEN 16 (NVIDIA GeForce RTX 4060 Laptop GPU 8GB, Intel Core i7-13620H, 16GB DDR5, Windows 11 + WSL2)

---

## 1. Objectives & Scope
M00 establishes the hardware environment, WSL2 configuration, Python virtual environment with lockfile pinning, core shared utilities (`ares.common`, `ares.common.stats`), automated integrity gates (blocking manuscript target numbers, silent fallbacks, dirty worktrees), job scheduling and telemetry infrastructure, and VRAM capacity benchmarking. No scientific claim from the paper is evaluated in M00; all subsequent milestones depend on M00's infrastructure.

---

## 2. Pre-Registered Hypotheses & Test Criteria

### 2.1 Determinism and Kill-and-Resume
* **Hypothesis 1 (Determinism):** A seeded forward pass with BERT-base run twice under `torch.use_deterministic_algorithms(True, warn_only=True)` and fixed seeds will produce bit-for-bit identical output logits.
* **Hypothesis 2 (Resumable Training):** A training run of DeBERTa-v3-base for 300 steps on dummy sequence classification, killed abruptly via `SIGKILL` at step 150 and resumed from the atomic checkpoint, will converge to a final loss matching an uninterrupted run within $|\Delta\text{loss}| \le 1.0 \times 10^{-3}$.

### 2.2 Statistical Correctness
* **Hypothesis 3 (Wilcoxon Exact Minimum $p$):** 
  * For $K = 5$ paired observations, the minimum achievable two-sided exact Wilcoxon signed-rank $p$-value is $2^{-4} = 0.0625$. Any claim of $p < 0.01$ at $K=5$ is mathematically impossible.
  * For $K = 10$, the minimum achievable two-sided exact $p$-value is $2 \times (1/2^{10}) = 0.001953125$.
* **Hypothesis 4 (Independent Metric Auditor):** A standalone, hand-written macro-F1 function will match `sklearn.metrics.f1_score(average="macro")` with $|\Delta\text{F1}| < 10^{-12}$ on 1,000 synthetic multi-class trials.
* **Hypothesis 5 (Bootstrap CI Coverage):** A percentile bootstrap CI ($B = 10,000$) on Bernoulli trials with known probability $p=0.5$ at $N=1,000$ will cover the true parameter with nominal 95% coverage across repeated simulations.

### 2.3 Integrity Guardrails
* **Test 1 (`test_no_manuscript_numbers.py`):** The repository scanner must reject any unannotated literal occurrence of the 25 audited paper numbers in `src/`, `configs/`, or `scripts/`. Injected mutation of `0.893` must cause test failure.
* **Test 2 (`test_no_silent_fallback.py`):** AST scanner must detect and fail on any bare `except:` or `except Exception: pass`, or hardcoded fallback data literals in dataset modules.
* **Test 3 (`test_results_schema.py`):** Every result JSON emitted must conform to the Pydantic schema defined in Protocol §4.
* **Test 4 (`test_test_set_log.py`):** Every evaluation result on a test split must have a corresponding logged entry in `milestones/test_set_access_log.csv`.

### 2.4 Hardware & VRAM Feasibility
* **Target:** 8 GB VRAM budget on RTX 4060 Laptop.
* **Verification:** DeBERTa-v3-base at sequence lengths 128, 256, 512 using mixed precision (bf16) with micro-batch sizes {8, 16, 32} must fit within 8,188 MiB VRAM without out-of-memory errors. Gradient accumulation must be configured to achieve declared effective batch size of 32.

---

## 3. Decision Rules
1. M00 cannot close if any unit test or integrity test fails.
2. M00 cannot close if the kill-and-resume discrepancy exceeds $|\Delta\text{loss}| > 1.0 \times 10^{-3}$.
3. All restricted dataset access requests must be formally documented in `access_requests.md` with explicit URLs and instructions for human submission.
