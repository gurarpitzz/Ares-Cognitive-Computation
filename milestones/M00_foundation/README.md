# M00 — Foundation, Integrity Infrastructure & Access Requests

**Status:** COMPLETE  
**Git tag:** `m00-complete`  
**PREREG commit:** `c0739c5`  

---

## 1. Summary
M00 establishes the hardware environment, WSL2 operational limits, repository scaffold, mathematical testing libraries, integrity guardrails, job queue, and automated ledger tooling for the ARES rebuild. All 18 unit and integrity tests pass with 100% test coverage. GPU determinism and atomic kill-and-resume tests achieved bit-exact reproducibility ($\Delta\text{loss} = 0.00\times 10^0$), and VRAM profiling verified that all sequence lengths (up to 512) and micro-batch sizes (up to 32) fit within the 8,188 MiB VRAM envelope of the RTX 4060 Laptop GPU.

---

## 2. Ledger Claims Addressed

| Ledger ID | Manuscript Claim | Manuscript Value | Measured Value (95% CI) | New Status | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C097** | 95% CIs computed by bootstrap resampling | Normal approx formula ($\bar{F}_1 \pm 1.96\sigma/\sqrt{n}$) | Non-parametric percentile bootstrap ($B=10,000$) implemented in `ares.common.stats.bootstrap_ci` | `REVISED` | `ares_v2/tests/common/test_stats.py` |

---

## 3. What Was Built

* `ares_v2/src/ares/common/paths.py`: Canonical path resolution for workspace, data, cache, and milestones.
* `ares_v2/src/ares/common/seed.py`: Global deterministic seed enforcement across CPU and CUDA backends.
* `ares_v2/src/ares/common/log.py`: Structured console and file logger.
* `ares_v2/src/ares/common/manifest.py`: SHA-256 streaming dataset and directory manifest hashing.
* `ares_v2/src/ares/common/results.py`: Pydantic result model enforcing Protocol §4 provenance and git dirty worktree blocking.
* `ares_v2/src/ares/common/stats.py`: Complete statistical testing suite (exact Wilcoxon, bootstrap CIs, McNemar, Holm-Bonferroni, ECE, Brier, and independent hand-written F1).
* `ares_v2/src/ares/common/telemetry.py`: GPU thermal throttle detector parsing 30-second interval telemetry logs.
* `ares_v2/src/ares/common/train_loop.py`: Resumable training loop wrapper with atomic `.tmp` checkpointing and full RNG restoration.
* `ares_v2/src/ares/common/track.py`: Lightweight file-backed experiment tracker.
* `ares_v2/scripts/ledger.py`: CLI tooling for claims ledger validation, delta merging, and evidence verification.
* `ares_v2/scripts/queue/{submit.sh, worker.sh, status.sh, gpu_telemetry.sh}`: FIFO GPU job scheduler and telemetry monitor.
* `ares_v2/scripts/backup.sh`: Secondary storage rsync backup to `E:\ares_backup`.
* `ares_v2/scripts/disk_report.sh`: Storage budget monitor enforcing $\le 150\text{ GB}$ on D:.
* `ares_v2/tests/integrity/test_no_manuscript_numbers.py`: AST/regex scanner blocking 25 audited paper constants from code/configs.
* `ares_v2/tests/integrity/test_no_silent_fallback.py`: AST scanner forbidding bare `except:` and silent exception `pass`.
* `ares_v2/tests/integrity/test_results_schema.py`: Schema validator for all results JSON files.
* `ares_v2/tests/integrity/test_test_set_log.py`: Access log validator for test set evaluations.
* `milestones/M00_foundation/access_requests.md`: Access request dossier for 5 Tier-1 and 10 Tier-2 restricted datasets.

---

## 4. Hardware & Environment Data

* **Host Machine:** HP OMEN Gaming Laptop 16-wd0xxx
* **Host OS:** Windows 11 Home 24H2 (Build 26200)
* **Linux Environment:** WSL2 Ubuntu 26.04 LTS (Kernel 6.18.33.1-microsoft-standard-WSL2)
* **Processor:** 13th Gen Intel Core i7-13620H (10 Cores, 16 Threads)
* **Graphics:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB GDDR6, Driver 592.82)
* **Memory Allocation:** 12 GB RAM + 16 GB swap file on `D:\wsl\swap.vhdx`
* **Python Runtime:** Python 3.11.17 via `uv`
* **Deep Learning Framework:** PyTorch 2.6.0 with CUDA 12.x / 13.x

---

## 5. Experiments & Benchmarks

| Experiment ID | Script | Seed | Hardware | Wall-clock (s) | Primary Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `M00-smoke-determinism` | `scripts/smoke/smoke_determinism.py` | 42 | RTX 4060 Laptop | 0.05s | $\Delta = 0.00\times 10^0$ (Bit-exact PASS) |
| `M00-smoke-kill-resume` | `scripts/smoke/smoke_kill_resume.py` | 42 | RTX 4060 Laptop | 3.8s | $\Delta\text{loss} = 0.00\times 10^0$ (Bit-exact PASS) |
| `M00-smoke-vram-fit` | `scripts/smoke/smoke_vram_fit.py` | 42 | RTX 4060 Laptop | 48.2s | 9/9 configurations PASS (Peak 8,032 MiB) |

---

## 6. Results

### VRAM Capacity & Throughput Suite (`milestones/M00_foundation/tables/vram_fit.csv`)

| Model Equivalent | Sequence Length | Micro-Batch | Effective Batch | Precision | Peak VRAM (MiB) | Steps / sec | Samples / sec | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DeBERTa-base (~110M) | 128 | 8 | 32 | bf16 | 2,168.2 | 7.52 | 60.2 | PASS |
| DeBERTa-base (~110M) | 128 | 16 | 32 | bf16 | 2,331.2 | 5.33 | 85.3 | PASS |
| DeBERTa-base (~110M) | 128 | 32 | 32 | bf16 | 3,145.7 | 3.03 | 97.1 | PASS |
| DeBERTa-base (~110M) | 256 | 8 | 32 | bf16 | 2,341.3 | 5.84 | 46.7 | PASS |
| DeBERTa-base (~110M) | 256 | 16 | 32 | bf16 | 3,148.4 | 3.28 | 52.4 | PASS |
| DeBERTa-base (~110M) | 256 | 32 | 32 | bf16 | 4,777.4 | 1.79 | 57.3 | PASS |
| DeBERTa-base (~110M) | 512 | 8 | 32 | bf16 | 3,143.7 | 3.16 | 25.3 | PASS |
| DeBERTa-base (~110M) | 512 | 16 | 32 | bf16 | 4,774.1 | 1.67 | 26.7 | PASS |
| DeBERTa-base (~110M) | 512 | 32 | 32 | bf16 | 8,032.7 | 0.49 | 15.8 | PASS |

---

## 7. Deviations from Plan / PREREG
None. All pre-registered hypotheses and test tolerances were satisfied without modification.

---

## 8. Negative or Unexpected Results
* **PyTorch 2.6 Default Serialization:** PyTorch 2.6 defaults `torch.load(..., weights_only=True)`. Attempting to restore Python/NumPy RNG states failed initially until `weights_only=False` was explicitly designated in `CheckpointManager.load_latest`.
* **RNG Reset Bug Caught:** In `smoke_kill_resume.py`, calling `create_synthetic_data` after `load_latest` inadvertently re-initialized RNG seeds. Reordering the loader instantiation before checkpoint restoration resolved the issue and yielded bit-exact identical loss ($0.722015$ vs $0.722015$).

---

## 9. Limitations & Threats to Validity
At sequence length 512 with micro-batch 32, peak VRAM allocated reached 8,032.7 MiB (98.1% of available 8,188 MiB VRAM). While it did not crash, it operates near the physical memory threshold. Consequently, for production training in M03, sequence length 512 must use micro-batch 8 or 16 with gradient accumulation (4 or 2 steps), keeping memory usage safely between 3.1 GB and 4.7 GB.

---

## 10. Manuscript Impact
* **Section XXII-B:** The formula stating $95\%\text{ CI} = \bar{F}_1 \pm 1.96\sigma/\sqrt{n}$ is formally categorized as an invalid normal approximation and replaced with non-parametric percentile bootstrap resampling ($B=10,000$). Ledger claim C097 status updated to `REVISED`.

---

## 11. How to Reproduce

Run the master M00 reproduction script:
```bash
export PYTHONPATH="ares_v2/src:$PYTHONPATH"
bash ares_v2/scripts/smoke/run_m00.sh
```
Expected runtime: ~75 seconds on HP OMEN 16 (RTX 4060 Laptop).

---

## 12. Audit Summary
* **Audit Document:** `milestones/M00_foundation/AUDIT.md`
* **Findings:** 0 BLOCKER, 0 MAJOR, 0 MINOR.
* **Verdict:** CLOSE
