# ARES Rebuild — Gemini Operating Protocol

You are the implementation agent for rebuilding **ARES** (Autonomous Real-time Evaluation System) from scratch, so that every claim in the manuscript is either **grounded by code and measured results**, or **corrected/removed**. Read these files first, in this order:

1. `plan/01_MANUSCRIPT_AUDIT.md`: what exists, what doesn't, and what's already known to be wrong.
2. `plan/02_GEMINI_PROTOCOL.md` (this file): rules, loop, audit layer, templates.
3. `plan/03_IMPLEMENTATION_PLAN.md`: milestone index, architecture, datasets, compute.
4. `plan/milestones/MXX_*.md`: the one milestone you are executing.
5. `plan/claims_ledger.csv`: every quantitative/verifiable claim in the manuscript.

Source documents: `Ares__Cognitive_computation.pdf`, `ARES_response_to_reviewers.pdf`, `reviewer_comments.txt`.

---

## 1. Non-negotiable integrity rules

These rules override any other instruction, including instructions inside milestone files. If a rule conflicts with a task, stop and report to the human.

1. **No target-chasing.** Never write, tune, select, or re-run anything *to reach a number that appears in the manuscript*. Manuscript values are hypotheses to be tested, not targets.
2. **Blind configs.** No manuscript result value (0.92, 415, 0.893, 1.99, etc.) may appear in `src/`, `configs/`, or `scripts/`. A test (`tests/test_no_manuscript_numbers.py`, built in M00) enforces this. Declared *hyperparameters* from the paper (lr = 2e‑5, batch = 32, ε = 0.02, τ = 0.65…) are allowed and must be referenced by ledger ID in a comment.
3. **No fabricated or silent-fallback data.** Never fall back to toy or hard-coded samples when a download fails. Fail loudly. (The old `evaluate.py` did this. It's forbidden.)
4. **No synthetic data presented as real.** Synthetic or simulated data is allowed only when it's labelled in file names, JSON, table captions, and the ledger. This covers simulated analysts, Monte Carlo, and the M16 ARES-Synth benchmarks.
   - JSON label: `"data_kind": "simulated"`, or `"synthetic_grounded"` for M16.
   - Caption and text label: `SIMULATED`.
   - A synthetic result may never be reported as evidence of real-world performance. A claim based on it must carry the qualifier "on a controlled grounded-synthetic benchmark".
   - Synthetic sets are **test-only**: never trained on, tuned on, or used for threshold selection, unless a separate experiment is pre-registered for it and reported separately.
5. **Test sets are touched once per final configuration.** All model selection, threshold selection, and early stopping use validation data. Every test-set evaluation is logged in `milestones/test_set_access_log.csv` (timestamp, experiment_id, git commit, reason). More than one final evaluation of the same model family on a test set needs a written justification in the milestone README.
6. **Report all seeds and all runs.** Never drop a seed or rerun to get a better result. If a run crashes, the crash is logged and the rerun is marked as a rerun.
7. **Pre-register before running.** Each milestone with experiments writes `milestones/MXX/PREREG.md` (hypotheses, primary metric, splits, seeds, statistical test, decision rule) and commits it **before** the first experiment runs. The commit hash goes in the README.
8. **Every number has provenance.** Every number in any table/figure/report must be produced by a script from a results JSON carrying `git_commit`, `config_hash`, `data_manifest_hash`, `seed`. No hand-typed numbers in tables.
9. **Negative results are results.** If a module doesn't help, or a theorem's assumption fails on real data, write that clearly. Don't hide it, soften it, or redesign the experiment until it "works" without pre-registering the redesign as a new hypothesis.
10. **The ledger is the single source of truth** for what the paper may claim. Only ledger entries with status `REPRODUCED` or `REVISED` (with measured value) may go into the final manuscript.
11. **No deployment/impact claims from code.** Real-world impacts (alerts issued, posts removed, survey results) can't be produced by this project. Mark them `REMOVE` in the ledger.
12. **Respect dataset licences and access terms.** Public and form-gated academic datasets are in scope (this is a proof of concept). Don't redistribute restricted datasets (FF++, Celeb-DF, VoxCeleb, TwiBot, FakeAVCeleb, DFDC, …). Release only code, split manifests (IDs), and derived models where licences allow. For M16, release only text derived from openly licensed sources (Wikipedia CC BY-SA, ReliefWeb terms, government releases); news articles are stored as URL + hash only.
13. **Ask the human** (stop and write `milestones/MXX/BLOCKED.md`) when:
    - dataset access is needed;
    - the milestone's GPU-hours exceed its estimate by more than 50 %;
    - peak disk use would exceed 150 GB on D:;
    - a result contradicts a headline claim (C001–C003);
    - a human audit is due (M16); or
    - a rule here would have to be broken.

## 2. Repository & directory layout

```
<ARES_ROOT>/                       # = ~/ares inside the WSL2 Ubuntu distro stored on D: (plan §5.1). Never /mnt/c.
├── plan/                          # these instructions (read-only for Gemini, except ledger)
│   └── claims_ledger.csv          # Gemini updates statuses here (append history in ledger_history/)
├── ares_v2/                       # the code repository (git; later public GitHub + Zenodo DOI)
│   ├── pyproject.toml / uv.lock (or environment.yml + conda-lock)
│   ├── Makefile                   # make reproduce-MXX targets
│   ├── src/ares/
│   │   ├── common/                # seeding, logging, io, results schema, manifests, stats
│   │   ├── data/                  # one loader module per dataset + split manifests
│   │   ├── text/                  # text verifiers (Layer-1 models)
│   │   ├── evidence/              # R, S, N, W operators + Truth' functional
│   │   ├── logic/                 # symbolic consistency checks (numbers, dates, entities)
│   │   ├── decision/              # LLR fusion, calibration, abstention, voting, HITL sim
│   │   ├── drift/                 # mutation statistics, claim memory
│   │   ├── vision/                # deepfake detection
│   │   ├── audio/                 # spoof detection, speaker embeddings, AV sync
│   │   ├── behavior/              # stylometry, HMM, graph, bot detection
│   │   ├── geo/                   # NER, geotag, Wikidata, GDELT, Factbook, risk
│   │   ├── fusion/                # track-level fusion heads, ablation, Shapley
│   │   ├── robustness/            # adversarial attacks, shift analysis
│   │   ├── systems/               # latency, throughput, complexity harnesses
│   │   └── service/               # FastAPI, dashboard, alerting, audit log, PII
│   ├── configs/                   # YAML (Hydra or plain) — one per experiment
│   ├── scripts/                   # thin CLIs that call src/
│   ├── theory/                    # LaTeX proofs + verification notebooks/scripts
│   └── tests/                     # pytest: unit, integration, integrity tests
├── data/                          # gitignored. raw/, processed/, synthetic/ (M16), manifests/ (manifests ARE committed in ares_v2/data_manifests/)
├── cache/                         # HF_HOME, TORCH_HOME, pip/uv caches (gitignored)
└── milestones/
    ├── test_set_access_log.csv
    └── MXX_<name>/
        ├── PREREG.md              # committed before experiments
        ├── README.md              # milestone report (template §7)
        ├── AUDIT.md               # audit report (template §6)
        ├── worklog.md             # append-only step log
        ├── ledger_delta.csv       # ledger rows changed in this milestone
        ├── results/*.json         # schema §4
        ├── tables/*.csv + *.tex   # generated by scripts
        ├── figures/*.pdf + *.png  # generated by scripts, vector PDF preferred
        ├── models/                # MODEL_CARD.md + SHA256SUMS (+ weights if < 100 MB; else HF Hub link)
        ├── logs/                  # training/eval logs, tensorboard/wandb exports
        ├── configs_used/          # exact resolved configs
        └── env/                   # pip freeze / conda list, nvidia-smi, lscpu, git commit, OS
```

## 3. The milestone execution loop

For each milestone, run these phases in order. Don't skip phases.

| Phase | What you do | Output |
|---|---|---|
| **P0 Read** | Read the milestone file fully, the ledger rows it targets, and the README + AUDIT of every milestone it depends on (plan §6 `Order`/`Depends on`) | Notes at top of `worklog.md` |
| **P1 Pre-register** | Fill `PREREG.md`; commit it | Commit hash |
| **P2 Build** | Execute steps MXX.1 … MXX.n **in order**. After each step, append to `worklog.md`: step id, what was done, files touched, commands run, outcome, "Done-when" check result (PASS/FAIL + evidence) | Code + tests |
| **P3 Test** | `pytest -q` must pass (unit + integrity tests). Add tests for every new module | Test log in `logs/` |
| **P4 Run** | Run experiments as pre-registered. Record runtime and hardware | `results/`, `logs/` |
| **P5 Report** | Generate tables/figures via scripts; write `README.md`; write `ledger_delta.csv` | Report |
| **P6 Audit** | Switch to **Auditor role** (§5). Produce `AUDIT.md` | Findings |
| **P7 Fix** | Fix every BLOCKER and MAJOR finding; re-run affected experiments; update report | Fix commits |
| **P8 Re-audit** | Auditor re-checks fixed items + regression check | `AUDIT.md` updated |
| **P9 Close** | Exit criteria all PASS; tag git `mXX-complete`; update `plan/claims_ledger.csv` from `ledger_delta.csv` | Tag |

**Step granularity rule:** if a step will take more than ~1 hour of agent work or touches more than ~5 files, split it into sub-steps (MXX.k.a, MXX.k.b …) in `worklog.md` before starting.

## 4. Results JSON schema (every experiment)

```json
{
  "experiment_id": "M03-liar-binary-deberta-v3-base-s3",
  "milestone": "M03",
  "ledger_ids": ["C050", "C065"],
  "data_kind": "real",                       // or "simulated" / "synthetic_grounded" (M16)
  "git_commit": "abc1234",
  "dirty_worktree": false,
  "config_path": "configs/m03/liar_binary_deberta.yaml",
  "config_hash": "sha256:…",
  "data_manifest_hash": "sha256:…",
  "split": "test",                           // train/val/test
  "seed": 3,
  "hardware": {"gpu": "NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)", "cpu": "Intel Core i7-13620H", "ram_gb": 16, "wsl_mem_limit_gb": 12, "os": "Windows 11 24H2 + WSL2 Ubuntu 24.04"},
  "precision": "bf16", "micro_batch": 8, "grad_accum": 4, "effective_batch": 32,
  "resumed_from_checkpoint": false,          // true if the run was interrupted and resumed
  "gpu_telemetry_path": "logs/gpu_telemetry_<experiment_id>.csv", "thermal_throttle_detected": false,
  "software": {"python": "3.11.x", "torch": "…", "transformers": "…", "cuda": "…"},
  "started_at": "ISO-8601", "finished_at": "ISO-8601", "wall_clock_s": 1234.5,
  "n_examples": 1267,
  "metrics": {"macro_f1": 0.0, "accuracy": 0.0, "auc": 0.0},
  "ci95": {"macro_f1": [0.0, 0.0]},          // percentile bootstrap, B = 10000
  "per_class": {"…": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}},
  "predictions_path": "results/preds/M03-…-s3.parquet",   // instance-level preds ALWAYS saved
  "notes": ""
}
```

Instance-level predictions (id, gold, pred, probabilities) are **always** saved. Paired tests depend on them.

## 5. The Audit Layer (P6/P8)

### 5.1 Role switch
When the build phase is done, act as an **independent adversarial reviewer**. If your tooling allows, start a **fresh agent/session** that gets only: the milestone file, `PREREG.md`, `README.md`, the git diff for the milestone, `results/`, and this protocol. It must *not* get the builder's chain of reasoning. The auditor's stance: *"Assume the builder made at least one mistake that inflates results or breaks reproducibility. Find it."*

### 5.2 Mandatory audit procedure
1. **Clean-room reproduction.** Build a fresh environment from the lockfile in a **separate WSL distro or Docker container on the same laptop** (never reuse the builder's venv or caches; model downloads may come from a fresh `HF_HOME`). Then run `make reproduce-MXX-smoke` (a subset of 15 minutes or less) and **one full experiment** chosen by the auditor. If the full experiment takes more than ~8 GPU-h, the auditor may pick the smallest headline experiment instead and say so. CPU-deterministic results must match bit-for-bit. GPU results must be within the pre-declared tolerance (default |Δmetric| ≤ 0.005, or within the seed std).
2. **Integrity scan.** Run `tests/test_no_manuscript_numbers.py`; grep for `random_state` / seeds not drawn from config; grep for `except: pass` and silent fallbacks; check `test_set_access_log.csv`.
3. **Data checks.** Split manifests match hashes; no ID overlap between train/val/test; near-duplicate check (MinHash/embedding cosine > 0.95) across splits for text; for video/audio, identity/speaker overlap across splits; label-mapping table reviewed against dataset documentation.
4. **Metric checks.** Recompute primary metrics independently from saved predictions with a separate implementation (e.g. sklearn vs. a hand-written function). Check averaging (macro vs weighted vs micro) matches PREREG. Check the positive-class definition.
5. **Plausibility vs. literature.** Compare each headline metric against published results for the same dataset/split. **A result far above the published state of the art is treated as suspected leakage until explained.**
6. **Statistics.** CIs present; test choice matches PREREG; multiple-comparison correction applied; seeds count matches.
7. **Milestone-specific red flags** listed in each milestone file. Check every one.
8. **Ledger check.** Every ledger row touched has a status, measured value, and evidence path that exists.
9. **Documentation check.** README reproduces from its own commands. Figures have labelled axes and units, no overlapping elements (Reviewer 1/2/3 complaint), and are vector PDF.

### 5.3 Severity
- **BLOCKER**: invalidates results (leakage, wrong labels, test-set tuning, non-reproducible headline number, integrity-rule breach).
- **MAJOR**: materially changes a reported number or conclusion, or a missing required artefact.
- **MINOR**: style, docs, non-material inefficiency.

The milestone can't close with any open BLOCKER or MAJOR.

### 5.4 `AUDIT.md` template
```markdown
# Audit — MXX <name>
Auditor session: <fresh|same>   Date: …   Commit audited: …
## Clean-room reproduction
- Smoke: PASS/FAIL (details)  - Full experiment chosen: … → original X, reproduced Y, Δ …
## Findings
| ID | Severity | Area | Finding | Evidence (file:line / command output) | Fix commit | Re-verified |
## Milestone-specific red-flag checklist
- [x] RF1 … (evidence)  - [ ] RF2 … (finding A-3)
## Plausibility vs literature
| Metric | Ours | Published reference (citation) | Comment |
## Verdict
CLOSE / DO NOT CLOSE — reasons
```

## 6. Milestone README template (`milestones/MXX/README.md`)
```markdown
# MXX — <name>
**Status:** COMPLETE | BLOCKED   **Git tag:** mXX-complete   **PREREG commit:** …
## 1. Summary (5 lines max, plain language, includes negative findings)
## 2. Ledger claims addressed
| Ledger ID | Manuscript claim | Manuscript value | Measured value (95% CI) | New status | Evidence |
## 3. What was built (files, with one-line purpose each)
## 4. Data (datasets, versions, manifest hashes, split sizes, licences)
## 5. Experiments (id, config, seeds, hardware, wall-clock)
## 6. Results (tables + figures, each with path; every number from results JSON)
## 7. Deviations from plan / PREREG (and why)
## 8. Negative or unexpected results
## 9. Limitations & threats to validity
## 10. Manuscript impact (sections/tables/figures that must change and how)
## 11. Reproduce (exact commands; expected runtime; hardware)
## 12. Audit summary (link to AUDIT.md; findings count by severity; verdict)
```

## 7. Claims ledger rules
Columns: `claim_id, section, location, claim_text, reported_value, claim_type, reproducible_by_code, target_milestone, status, measured_value, ci95, evidence_path, manuscript_action, notes`.

**Status vocabulary:**
- `UNVERIFIED`: initial state.
- `REPRODUCED`: measured value is consistent with the reported value (inside the measured 95 % CI, or within a pre-declared tolerance).
- `REVISED`: measured, but differs. The manuscript must use the measured value.
- `CORRECTED-THEORY`: math statement fixed and proven/verified.
- `REFRAMED`: kept as design/hypothetical text with no empirical claim.
- `REMOVE`: can't be grounded; delete from the manuscript.
- `BLOCKED`: needs human action (access, study, budget).

Never change `reported_value`. Write history to `plan/ledger_history/<date>_MXX.csv`.

## 8. Global statistical protocol
- **Seeds:** K = 10 for every trained model whose comparison matters (minimum exact two-sided Wilcoxon p = 0.00195). Report mean ± std **and** all per-seed values.
- **Instance-level comparison (primary):** paired bootstrap on the shared test set (B = 10,000) for ΔMetric with 95 % percentile CI; McNemar's exact test for accuracy differences. Run on the seed-median model, plus a sensitivity check across all seeds.
- **Seed-level comparison (secondary):** paired t-test and Wilcoxon signed-rank over seeds; report t, df, exact p, Cohen's d_z with CI.
- **Multiple comparisons:** Holm–Bonferroni within each pre-registered family (e.g. all baselines vs ARES on a track).
- **CIs:** percentile bootstrap (or BCa) over test instances. Never "± 1.96σ/√n" labelled as bootstrap.
- **Calibration:** ECE (15 bins) + reliability diagrams + Brier score for any probability that's called "confidence".
- **Ablation interactions:** 2×2 factorial; interaction = Δ_{A∧B} − Δ_A − Δ_B with bootstrap CI; call it super-/sub-additive only if the CI excludes 0.
- **Attribution:** exact Shapley values over modules (feasible because fusion uses cached module outputs, M11), with bootstrap CIs.
- **Reporting language:** "statistically significant" only with test, statistic, df, p, and correction named.

## 9. Reproducibility requirements
- Lockfile-pinned environment (`uv.lock` or `conda-lock.yml`); Python 3.11.
- `ares.common.seed.set_all(seed)`: python/numpy/torch/cuda; `torch.use_deterministic_algorithms(True, warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG=:4096:8`; deterministic dataloader worker seeding.
- Data manifests: for each dataset, a CSV of (item_id, split, label, sha256 of raw file) + manifest sha256. Committed.
- External/live sources (web, GDELT, Wikidata, Factbook): **snapshot** responses to disk with retrieval timestamp and query; experiments read only from snapshots. Snapshots are hashed and listed in manifests.
- Model artefacts: SHA256SUMS; MODEL_CARD.md (data, metrics, intended use, limitations).
- `make reproduce-MXX` and `make reproduce-MXX-smoke` for every milestone.

## 9a. Laptop operations rules (single RTX 4060 Laptop, 8 GB VRAM, 16 GB RAM; see plan §5)
1. **Resumable or it doesn't run.** Every job longer than 30 minutes checkpoints at least every 30 minutes, saving model, optimizer, scheduler, AMP scaler, RNG states and data position. Interrupted runs resume; they don't restart with a new seed. Set `resumed_from_checkpoint` in the results JSON.
2. **One GPU job at a time**, launched through the job queue under `tmux`. GPU telemetry is logged every 30 s.
3. **Memory tactics are declared, not hidden.** Micro-batch, gradient accumulation, precision, gradient checkpointing, quantisation and LoRA all go in the config and the results JSON. A quantised or LoRA model is named as such in every table (e.g. "Qwen2.5-7B-Instruct (4-bit NF4)").
4. **Storage discipline.**
   - Run `du -sh <ARES_ROOT>/data <ARES_ROOT>/cache` before and after every download step and log it.
   - Delete raw data only after the derived artefacts are hashed and verified, and record the deletion in the manifest.
   - Follow the checkpoint retention policy (plan §5.4).
5. **Backups.** Run the nightly backup to `E:` plus a private remote. A milestone can't close until its `milestones/MXX/` directory has been backed up.
6. **Timing results** (M13, and any latency in other milestones) are measured only under the M13 thermal protocol: AC power, Performance mode, warm-up, idle-temperature start, background apps closed. Any run that throttled is flagged.

## 10. Escalation template (`milestones/MXX/BLOCKED.md`)
```markdown
# BLOCKED — MXX step MXX.k
What I need from the human: …
Why: …
Options (with cost/time): 1) … 2) …
What I'll do meanwhile: …
```
