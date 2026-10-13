# M00 — Foundation, Integrity Infrastructure & Access Requests

**Objective:** Set up the laptop (WSL2 on D:, CUDA, caches, power/thermals, job queue, backups). Create the repository, environment, shared utilities (seeding, results schema, manifests, statistics, resumable trainer), integrity tests and ledger tooling. Measure what fits in 8 GB VRAM, and submit all restricted-dataset access requests. Nothing scientific is run here, but everything after depends on it.
**Ledger IDs:** all (ledger tooling), C097 (bootstrap CI method).
**Depends on:** —
**Compute:** The RTX 4060 Laptop (all of M00 runs on it, under WSL2). About 2 GPU-h for the smoke and fit tests.
**Output dir:** `milestones/M00_foundation/`

---

## Steps

### M00.0 — Laptop & WSL2 setup (do this first; HUMAN assists with Windows-side steps)
- **Action (Windows side):**
  1. Update the NVIDIA Windows driver (Game Ready or Studio; the current one is 592.82). Don't install any Linux NVIDIA driver inside WSL.
  2. `wsl --install -d Ubuntu-24.04`. Then move it to D:: `wsl --export Ubuntu-24.04 D:\wsl\ubuntu.tar` → `wsl --unregister Ubuntu-24.04` → `wsl --import Ubuntu-ARES D:\wsl\ares D:\wsl\ubuntu.tar`, then set the default user.
  3. Write `%UserProfile%\.wslconfig` with `[wsl2] memory=12GB processors=14 swap=16GB swapFile=D:\\wsl\\swap.vhdx`, then `wsl --shutdown`.
  4. Power: OMEN Gaming Hub → Performance. In Windows power settings, "Sleep: Never" and "Hibernate: Never" when plugged in. Set Windows Update active hours, and pause updates during multi-day runs. Turn on battery-care charge limiting if the BIOS/OMEN Hub offers it (battery health is 82.7 %).
  5. Create `E:\ares_backup\`.
- **Action (WSL side):**
  1. `nvidia-smi` must show the RTX 4060 Laptop GPU with 8188 MiB.
  2. Install `build-essential git git-lfs tmux task-spooler openjdk-21-jdk-headless ffmpeg rsync`.
  3. Create `<ARES_ROOT>=~/ares`. Export `ARES_ROOT`, `ARES_DATA_ROOT=$ARES_ROOT/data`, `HF_HOME=$ARES_ROOT/cache/hf`, `TORCH_HOME`, `PIP_CACHE_DIR` and `UV_CACHE_DIR` under `$ARES_ROOT/cache` in `~/.bashrc`.
  4. `df -h ~` must show the VHDX on D: with ≥ 150 GB available to grow into.
- **Files:** `docs/laptop_setup.md` (every command, screenshots optional), `milestones/M00_foundation/env/{nvidia-smi.txt,lscpu.txt,free.txt,df.txt,wslconfig.txt}`.
- **Done when:**
  - `nvidia-smi` works inside WSL.
  - `free -g` shows about 12 GB.
  - The distro VHDX is on D:.
  - The cache env vars resolve under `$ARES_ROOT/cache`.
  - Everything is logged in the worklog.

### M00.1 — Repository scaffold
- **Action:** Create `<ARES_ROOT>/ares_v2/` (inside WSL, **not** under `/mnt/c` or `/mnt/d`) as a git repo with the layout in Protocol §2. Add:
  - `.gitignore` (data/, *.pt, *.safetensors, *.ckpt, *.gguf, cache/, .venv/, wandb/, mlruns/);
  - `LICENSE` (MIT for code);
  - `CITATION.cff` (placeholder authors/title);
  - `README.md` (one paragraph + "work in progress");
  - `.gitattributes` (`* text=auto eol=lf`), so files edited from Windows don't break the shell scripts.
- **Files:** as listed; empty `__init__.py` in each `src/ares/*` subpackage.
- **Done when:** `git log` shows initial commit; `tree -L 3 ares_v2` matches Protocol §2.

### M00.2 — Pinned environment
- **Action:** Create a Python 3.11 environment with `uv` inside WSL.
  - **Base deps:** numpy, scipy, pandas, pyarrow, scikit-learn, statsmodels, matplotlib, seaborn, pydantic, pyyaml, hydra-core (optional), tqdm, rich, pytest, ruff, torch (CUDA 12.x wheel), transformers, accelerate, peft, bitsandbytes, datasets, sentence-transformers, faiss-cpu, jsonschema, ijson.
  - Produce a lockfile.
  - **Optional groups:**
    - `[vision]` (timm, opencv-python-headless, facenet-pytorch or insightface)
    - `[audio]` (torchaudio, librosa, speechbrain)
    - `[graph]` (torch-geometric, networkx)
    - `[geo]` (spacy, geopy)
    - `[retrieval]` (pyserini; needs Java 21)
    - `[llm]` (llama-cpp-python built with CUDA, as an alternative to bitsandbytes)
    - `[attack]` (textattack)
    - `[service]` (fastapi, uvicorn, presidio-analyzer, locust)
- **Command:** `make env` → creates env from lockfile; `make env-freeze` → writes `milestones/M00_foundation/env/`.
- **Done when:**
  - A fresh clone + `make env` inside WSL succeeds.
  - `python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"` prints the RTX 4060 Laptop and `True`.
  - The lockfile is committed.

### M00.3 — `ares.common` core utilities
- **Action:** Implement:
  - `seed.py`: `set_all(seed)` (python, numpy, torch, cuda, deterministic flags, `CUBLAS_WORKSPACE_CONFIG`), `worker_init_fn`.
  - `results.py`: pydantic model for the results JSON schema (Protocol §4); `write_result()` auto-fills git commit, dirty flag, hardware, software versions, timestamps; refuses to write if `dirty_worktree` and `--allow-dirty` not set.
  - `manifest.py`: `sha256_file`, `sha256_dir`, `build_manifest(df, cols)`, `manifest_hash()`.
  - `paths.py`: single source of paths (DATA_ROOT, MILESTONES_ROOT) read from env var `ARES_DATA_ROOT` with defaults.
  - `log.py`: structured logging to console + file.
- **Files:** `src/ares/common/{seed,results,manifest,paths,log}.py`; tests in `tests/common/`.
- **Done when:** unit tests pass; writing a dummy result produces valid JSON with all fields.

### M00.4 — `ares.common.stats` (statistical library)
- **Action:** Implement and unit-test:
  - `bootstrap_ci(metric_fn, y_true, y_pred, B=10000, method="percentile"|"bca", seed)`.
  - `paired_bootstrap_delta(metric_fn, y_true, pred_a, pred_b, B, seed)` → Δ, CI, one-sided/two-sided p.
  - `mcnemar_exact(correct_a, correct_b)`.
  - `paired_t(a, b)`, `wilcoxon_exact(a, b)` (scipy, exact mode), `cohens_dz(a, b)` with CI (noncentral-t method or bootstrap).
  - `holm_bonferroni(pvals)`.
  - `ece(probs, labels, n_bins=15)`, `brier()`, `reliability_curve()`.
  - `interaction_ci(metric_fn, preds_full, preds_minusA, preds_minusB, preds_minusAB, y, B)`.
  - `macro_f1`, `per_class_prf` wrappers (sklearn) **plus** an independent hand-written F1 for auditor cross-checks.
- **Tests (must include):** Wilcoxon exact two-sided min p for K = 5 equals 0.0625 and for K = 10 equals 0.001953125; bootstrap CI on a known binomial roughly matches Wilson interval; Holm on a textbook example; hand-written F1 == sklearn F1 on 1000 random cases.
- **Done when:** `pytest tests/common/test_stats.py` passes.

### M00.5 — Integrity tests
- **Action:** Implement:
  - `tests/integrity/test_no_manuscript_numbers.py`: loads a curated list of *distinctive* manuscript result values (e.g. `0.893, 0.875, 0.911, 0.948, 415, 1.99, 0.071, 0.058, 0.043, 0.031, 0.049, 0.112, 0.094, 0.036, 0.061, 6.8, 18.6, 1.26, 0.4231, 0.8799, 30000, 6.3, 22.4, 4.8, 88`) from `tests/integrity/manuscript_values.yaml`, scans `src/`, `configs/`, `scripts/` for literal occurrences, and fails unless the line carries an allow-comment `# ledger:CXXX hyperparameter`. (Values that are declared *hyperparameters*, like 2e-5 or 0.65, are not in the list.)
  - `tests/integrity/test_no_silent_fallback.py`: AST scan for bare `except:` / `except Exception: pass` and for hard-coded DataFrame literals in data loaders.
  - `tests/integrity/test_results_schema.py`: validates every JSON under `milestones/*/results/`.
  - `tests/integrity/test_test_set_log.py`: every results JSON with `split=="test"` has a matching row in `milestones/test_set_access_log.csv`.
- **Mutation check:** temporarily insert `0.893` into a config, confirm the test fails, revert. Record in worklog.
- **Done when:** all integrity tests pass on a clean tree and demonstrably fail on injected violations.

### M00.6 — Ledger tooling
- **Action:** `scripts/ledger.py` with subcommands: `validate` (schema, allowed statuses, IDs unique), `apply-delta <milestone>` (merges `ledger_delta.csv`, writes snapshot to `plan/ledger_history/`), `summary` (counts by status/milestone → markdown), `check-evidence` (every non-UNVERIFIED row has an existing `evidence_path`).
- **Done when:** `python scripts/ledger.py validate plan/claims_ledger.csv` passes; summary prints 99 UNVERIFIED rows.

### M00.7 — Makefile, CI, templates
- **Action:** Makefile targets: `env`, `test`, `lint`, `reproduce-MXX`, `reproduce-MXX-smoke` (stubs), `ledger-summary`. GitHub Actions workflow: ruff + pytest (CPU). Copy PREREG/README/AUDIT templates (Protocol §5.4, §6) into `milestones/_templates/`. Create `milestones/test_set_access_log.csv` with header.
- **Done when:** `make test lint` pass locally; CI file present.

### M00.8 — Restricted dataset access requests (HUMAN ACTION)
- **Action:** Write `milestones/M00_foundation/access_requests.md` with one row per restricted asset (Tier 1 and Tier 2 of plan §4). For each asset give: official URL, what to submit, licence summary, expected wait, and status.
  - **Tier 1:** FaceForensics++, Celeb-DF v2, VoxCeleb1 (test), TwiBot-20, MultiClaim.
  - **Tier 2:** FakeAVCeleb, NewsCLIPpings/VisualNews, DFDC preview, WildDeepfake, DeeperForensics-1.0, Snopes corpus (UKP), IFND, FactDrill, MOCHEG, Factify 2, VERITE, TwiBot-22.
  - **API keys and accounts:** Google Fact Check Tools API key, a web search API key (showcase/T11 only), ReliefWeb API appname, Hugging Face account/token (accept the model licences for Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3).
  - Then write `BLOCKED.md` asking the human to submit them **today**, and continue with the other steps.
- **Done when:** file exists; human confirms submission (status column updated).

### M00.9 — Laptop job infrastructure, resumable training, VRAM fit tests
- **Action:**
  1. **Job queue:** `scripts/queue/{submit.sh,worker.sh,status.sh}`. This is a file-based FIFO (or a `task-spooler` wrapper) run in a `tmux` session `ares-gpu`. It allows one GPU job at a time, and each job's stdout/stderr goes to `logs/`.
  2. **Telemetry:** `scripts/queue/gpu_telemetry.sh <exp_id>` runs the `nvidia-smi --query-gpu=… -l 30` logger. It starts and stops with each job. `ares.common.telemetry.detect_throttle(csv)` sets `thermal_throttle_detected` when the SM clock stays below 80 % of the run median while temperature is ≥ 87 °C for more than 5 minutes (thresholds go in config).
  3. **Resumable trainer:** `ares.common.train_loop`, a thin wrapper used by every training script. It checkpoints every N steps or every 30 minutes, whichever comes first. Each checkpoint holds model, optimizer, scheduler, AMP scaler, python/numpy/torch/cuda RNG states, epoch, and sampler position, written atomically as tmp + rename. `--resume auto` picks the latest checkpoint.
  4. **Backups:** `scripts/backup.sh` rsyncs `milestones/`, `ares_v2/` and the kept models to `/mnt/e/ares_backup/`, then pushes to the private remote. Install a nightly cron entry inside WSL.
  5. **Disk monitor:** `scripts/disk_report.sh` prints `du -sh` of data/cache/models and the free space on D:. It refuses to start a download if the projected peak exceeds 150 GB.
  6. **Compute ledger:** `milestones/compute_ledger.csv` with columns date, milestone, experiment_id, GPU-h, peak VRAM GB, peak disk GB, throttled (y/n).
- **Smoke and fit tests** (`scripts/smoke/`; each writes a results JSON):
  - (a) A seeded matmul plus a BERT-base forward pass, run twice, to check determinism.
  - (b) A **kill-and-resume test**: train DeBERTa-v3-base for 300 steps on a dummy task; kill at step 150 and resume; compare it with an uninterrupted run. Final loss must agree within 1e-3 (bit-exact is not required on GPU; record the difference).
  - (c) **VRAM fit table**: peak memory and steps/s for:
    - DeBERTa-v3-base, RoBERTa-base and mDeBERTa-base at len 128/256/512, micro-batch {8,16,32}, bf16;
    - DeBERTa-v3-large with bf16 + gradient checkpointing + 8-bit AdamW at len 128/256, micro-batch {2,4,8};
    - Qwen2.5-7B-Instruct in 4-bit NF4 inference (tokens/s, peak VRAM);
    - EfficientNet-B4 at 380² and Xception at 299², micro-batch {8,16,32};
    - AASIST on 4 s audio.
  - (d) **Re-estimate plan §5.5** GPU-hours from the measured steps/s; write `milestones/M00_foundation/compute_estimate.md`.
- **Done when:**
  - The smoke JSONs, the VRAM fit table (`tables/vram_fit.csv`) and `compute_estimate.md` exist.
  - The kill-and-resume test passes.
  - One backup has run, and `E:` contains it.

### M00.10 — Legacy POC import
- **Action:** Copy the current `ARES_POC` code into `ares_v2/legacy/ares_poc/` (read-only reference, with source commit SHA). Write `legacy/ares_poc/LEGACY_NOTES.md` listing its behaviour and flaws: live web retrieval (non-reproducible), silent toy fallbacks in `evaluate.py`, weighted-F1 over a 3-class mapping, keyword credibility, hard-coded −0.4 threshold, unbounded confidence, CPU-only NLI. Legacy code isn't imported by `src/`.
- **Done when:** notes committed.

### M00.11 — Experiment tracking
- **Action:** Pick MLflow (local file store) or W&B offline mode. Wire a thin wrapper `ares.common.track`. Results JSON stays canonical; tracking is a convenience only.
- **Done when:** a dummy run logs and the JSON is written.

---

## Audit gate — milestone-specific red flags
- RF1: Lockfile reproduces on a clean environment (auditor creates a new env from scratch in a second WSL distro or a container).
- RF2: Stats functions validated against scipy/statsmodels and analytic cases; Wilcoxon K = 5 min-p test present.
- RF3: Integrity tests actually fail on injected violations (mutation evidence in worklog).
- RF4: `write_result()` refuses dirty worktrees by default.
- RF5: No dataset content committed to git; `.gitignore` effective (`git check-ignore data/`).
- RF6: Access requests file complete and accurate (URLs resolve).
- RF7: The repo, data and caches live inside the WSL ext4 VHDX on D: (not under `/mnt/c`). The C: drive's free space has not dropped.
- RF8: The kill-and-resume test really kills the process (SIGKILL, not a graceful exit). Resumed RNG state is verified.
- RF9: The VRAM fit table comes from real runs (peak memory from `torch.cuda.max_memory_allocated` plus `nvidia-smi`), not from estimates.

## Exit criteria
- [ ] Laptop/WSL2 setup documented. GPU visible in WSL. VHDX on D:. Caches redirected. Power settings applied.
- [ ] Repo + env + lockfile committed; `make test lint` pass.
- [ ] `ares.common` + `ares.common.stats` + resumable `train_loop` complete and tested.
- [ ] Integrity tests pass and are mutation-verified.
- [ ] Ledger tooling works on `plan/claims_ledger.csv`.
- [ ] Access requests submitted by the human (or BLOCKED.md acknowledged).
- [ ] Smoke JSONs, kill-and-resume PASS, `vram_fit.csv`, `compute_estimate.md` present. Backup verified on E:.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Git tag `m00-complete`.
