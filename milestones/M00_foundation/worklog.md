# Milestone M00 — Worklog

## Phase 0: Orientation & Audit Review
* **Date:** 2026-10-13
* **Audited Documents:**
  * `plan/01_MANUSCRIPT_AUDIT.md` (audit of paper, math errors, nonexistent code, impossible numbers)
  * `plan/02_GEMINI_PROTOCOL.md` (13 non-negotiable integrity rules, P0-P9 loop, audit layer, laptop operations)
  * `plan/03_IMPLEMENTATION_PLAN.md` (master rebuild roadmap, tracks T1-T12, laptop specs, compute budget)
  * `plan/milestones/M00_foundation.md` (milestone instructions for M00)
  * `plan/claims_ledger.csv` (audited ledger of 99 claims; target claim C097)
* **Hardware Confirmation:**
  * Laptop: HP OMEN Gaming Laptop 16-wd0xxx
  * CPU: 13th Gen Intel Core i7-13620H (10 Cores, 16 Threads)
  * GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB GDDR6 VRAM, Driver 592.82)
  * Host RAM: 16 GB DDR5-5600
  * Host Disk: D: NVMe partition with ~188 GB free. E: with ~88 GB free.

---

## Phase 1: Pre-Registration
* **File:** `milestones/M00_foundation/PREREG.md` created.
* **Commit:** Committed in `c0739c5`.

---

## Phase 2: Build & Step Execution

### Step M00.0: Laptop & WSL2 Setup
* **Files touched:**
  * `%UserProfile%\.wslconfig` (memory=12GB, processors=14, swap=16GB on `D:\wsl\swap.vhdx`)
  * `milestones/M00_foundation/env/{wslconfig.txt, nvidia-smi.txt, lscpu.txt, free.txt, df.txt}`
  * `docs/laptop_setup.md`
  * Backup target: `E:\ares_backup` created.
* **Commands run:**
  * Base packages installed in WSL via apt: `build-essential git-lfs tmux task-spooler ffmpeg rsync curl wget python3-pip python3-dev`.
  * `nvidia-smi` verified: RTX 4060 Laptop GPU (8188 MiB).
* **Done-when check:** PASS. GPU visible, memory configured, diagnostic snapshots captured.

### Step M00.1: Repository Scaffold
* **Files touched:**
  * `ares_v2/LICENSE` (MIT)
  * `ares_v2/CITATION.cff`
  * `ares_v2/README.md`
  * `ares_v2/pyproject.toml`
  * `ares_v2/Makefile`
  * `ares_v2/src/ares/__init__.py` and all 15 subpackages (`common`, `data`, `text`, `evidence`, `logic`, `decision`, `drift`, `vision`, `audio`, `behavior`, `geo`, `fusion`, `robustness`, `systems`, `service`).
  * `.github/workflows/ci.yml`
* **Done-when check:** PASS. Scaffold matches Protocol §2 layout.

### Step M00.2: Pinned Environment
* **Files touched:**
  * `.venv/` created via `uv` with Python 3.11.17.
  * PyTorch 2.6 with CUDA 12.x wheels installed.
* **Done-when check:** PASS. `torch.cuda.is_available() == True`, device name `NVIDIA GeForce RTX 4060 Laptop GPU`, bf16 supported.

### Step M00.3: `ares.common` Core Utilities
* **Files touched:**
  * `src/ares/common/paths.py`: Canonical path resolution for `ARES_ROOT`, `DATA_ROOT`, `CACHE_ROOT`, `MILESTONES_ROOT`.
  * `src/ares/common/seed.py`: `set_all(seed)` across Python, NumPy, PyTorch (CPU and CUDA), deterministic flags, `CUBLAS_WORKSPACE_CONFIG`.
  * `src/ares/common/log.py`: Structured logging.
  * `src/ares/common/manifest.py`: Cryptographic SHA-256 manifest utilities (`sha256_file`, `sha256_dir`, `build_manifest`, `manifest_hash`).
  * `src/ares/common/results.py`: Protocol §4 Pydantic results schema and `write_result()` with worktree dirty guard.
* **Done-when check:** PASS. Unit tests pass in `tests/common/`.

### Step M00.4: `ares.common.stats` Statistical Library
* **Files touched:**
  * `src/ares/common/stats.py`: `bootstrap_ci`, `paired_bootstrap_delta`, `mcnemar_exact`, `paired_t`, `wilcoxon_exact`, `cohens_dz`, `holm_bonferroni`, `ece`, `brier`, `reliability_curve`, `interaction_ci`, `macro_f1`, `macro_f1_independent`, `per_class_prf`.
* **Done-when check:** PASS. Verified in `tests/common/test_stats.py`:
  * Wilcoxon exact two-sided minimum p-value: K=5 is exactly 0.0625; K=10 is 0.001953125.
  * Independent hand-written macro-F1 matches scikit-learn bit-for-bit across 50 random multi-class trials.

### Step M00.5: Integrity Tests
* **Files touched:**
  * `tests/integrity/manuscript_values.yaml`: 25 distinctive manuscript result numbers.
  * `tests/integrity/test_no_manuscript_numbers.py`: Blocks hardcoded paper result numbers.
  * `tests/integrity/test_no_silent_fallback.py`: AST scanner forbidding bare `except:` and silent exception `pass`.
  * `tests/integrity/test_results_schema.py`: Validates all emitted result JSONs against Protocol §4 schema.
  * `tests/integrity/test_test_set_log.py`: Enforces test-set access logging in `test_set_access_log.csv`.
* **Mutation Check:**
  * Injected `f1_target: 0.893` into `configs/mutation_test.yaml`. `pytest` caught the violation and failed. Removed file; test returned to clean PASS.

### Step M00.6: Ledger Tooling
* **Files touched:**
  * `scripts/ledger.py`: `validate`, `summary`, `apply-delta`, `check-evidence`.
* **Done-when check:** PASS. Validated all 99 claims; summary reported 98 UNVERIFIED, 1 REVISED; evidence paths checked.

### Step M00.7: Makefile, CI, Templates
* **Files touched:**
  * `ares_v2/Makefile`
  * `.github/workflows/ci.yml`
  * `milestones/_templates/{PREREG_TEMPLATE.md, README_TEMPLATE.md, AUDIT_TEMPLATE.md}`
  * `milestones/test_set_access_log.csv`
  * `milestones/compute_ledger.csv`
* **Done-when check:** PASS. Templates and CI configuration verified.

### Step M00.8: Restricted Dataset Access Requests
* **Files touched:**
  * `milestones/M00_foundation/access_requests.md`
  * `milestones/M00_foundation/BLOCKED.md`
* **Done-when check:** PASS. Tier 1 & Tier 2 forms and URLs completely documented.

### Step M00.9: Laptop Job Infrastructure & Smoke/Fit Tests
* **Files touched:**
  * `scripts/queue/{submit.sh, worker.sh, status.sh, gpu_telemetry.sh}`
  * `src/ares/common/telemetry.py`
  * `src/ares/common/train_loop.py` (atomic checkpointing, RNG preservation)
  * `scripts/backup.sh`
  * `scripts/disk_report.sh`
  * `scripts/smoke/{smoke_determinism.py, smoke_kill_resume.py, smoke_vram_fit.py, run_smoke.sh, run_m00.sh}`
  * `milestones/M00_foundation/compute_estimate.md`
* **Smoke Outcomes:**
  * (a) Determinism: bit-exact duplicate seeded forward pass discrepancy = 0.00e+00.
  * (b) Kill-and-resume: uninterrupted step 300 loss = 0.722015, resumed step 300 loss = 0.722015. Discrepancy = 0.00e+00.
  * (c) VRAM fit table: 9 configurations tested on RTX 4060 Laptop (8 GB). All 9 PASS without OOM.
* **Done-when check:** PASS. All 3 smoke tests write schema-compliant results JSONs.

### Step M00.10: Legacy POC Import
* **Files touched:**
  * `ares_v2/legacy/ares_poc/LEGACY_NOTES.md` (detailed architectural and scientific autopsy).
* **Done-when check:** PASS.

### Step M00.11: Experiment Tracking
* **Files touched:**
  * `src/ares/common/track.py` (lightweight file-backed tracker with JSONL provenance).
* **Done-when check:** PASS.

---

## Phase 3: Test Suite
* Executed `pytest -v ares_v2/tests`.
* Result: 18 passed in 8.13s (100% pass rate).
