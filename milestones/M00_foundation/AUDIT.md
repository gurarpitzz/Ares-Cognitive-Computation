# Adversarial Audit Report — M00 Foundation & Integrity Infrastructure

**Auditor Session:** Independent Adversarial Reviewer  
**Date:** 2026-10-13  
**Commit Audited:** HEAD (`c0739c5` + M00 implementation tree)  
**Target Milestone:** M00 (`M00_foundation.md`)  

---

## 1. Clean-Room Reproduction Audit

| Target | Command | Expected Outcome | Observed Outcome | Auditor Evaluation |
| :--- | :--- | :--- | :--- | :--- |
| **All Unit & Integrity Tests** | `pytest -v ares_v2/tests` | 18 passed in <= 15s | 18 passed in 8.13s | **PASS (Exact Match)** |
| **Smoke (a) Determinism** | `python ares_v2/scripts/smoke/smoke_determinism.py` | Max diff = 0.00e+00 | Max diff = 0.00e+00 | **PASS (Bit-exact)** |
| **Smoke (b) Kill-and-Resume** | `python ares_v2/scripts/smoke/smoke_kill_resume.py` | $\Delta\text{loss} \le 1.0\times 10^{-3}$ | $\Delta\text{loss} = 0.00\times 10^0$ | **PASS (Bit-exact)** |
| **Smoke (c) VRAM Fit** | `python ares_v2/scripts/smoke/smoke_vram_fit.py` | 9 configurations evaluated | 9/9 PASS, peak 8,032 MiB | **PASS (Real GPU telemetry)** |
| **Claims Ledger Tooling** | `python ares_v2/scripts/ledger.py validate` | All claims valid | 99 claims validated | **PASS** |
| **Ledger Delta Application** | `python ares_v2/scripts/ledger.py check-evidence` | Valid evidence paths | C097 verified | **PASS** |

---

## 2. Integrity Scan Audit

1. **Blind Configs / No Target Numbers:**
   - Executed `test_no_manuscript_numbers.py`: PASSED cleanly.
   - Mutation verification confirmed: Injecting `0.893` into `configs/mutation_test.yaml` resulted in immediate failure. Reverting cleared the failure.
2. **No Silent Fallback Data:**
   - Executed `test_no_silent_fallback.py`: PASSED cleanly.
   - AST inspection verified zero bare `except:` statements and zero silent exception `pass` statements in `src/`.
3. **Worktree Dirty Guard:**
   - `ares.common.results.write_result` inspected: explicitly calls `get_git_info()` and raises `RuntimeError` if git status is dirty unless `--allow-dirty` is provided.
4. **Git Tracking Discipline:**
   - Executed `git status`: `.gitignore` prevents tracking of `data/`, `cache/`, `*.pt`, `*.safetensors`, `.venv/`.

---

## 3. Milestone-Specific Red Flag Checklist

* [x] **RF1: Environment and Lockfile Reproducibility:** `uv` environment created on Python 3.11.17; dependencies verified with CUDA 12 support on the local RTX 4060 Laptop GPU.
* [x] **RF2: Statistical Exactness Validated:**
  - Tested exact Wilcoxon signed-rank minimum $p$-values: for $K=5$, minimum exact two-sided $p = 0.0625$; for $K=10$, minimum exact two-sided $p = 0.001953125$. Confirms the manuscript's claim of "$p < 0.01$ at $K=5$" was mathematically impossible.
  - Independent hand-written macro-F1 matched scikit-learn bit-for-bit across 50 random multi-class trials ($|\Delta\text{F1}| < 10^{-12}$).
* [x] **RF3: Mutation Check Active:** Verified in worklog and execution history; failure triggered when forbidden value injected.
* [x] **RF4: Results Schema Conformance:** Validated with Pydantic across all emitted JSON files.
* [x] **RF5: Dataset Leakage and Privacy:** No raw datasets stored in repo; `.gitattributes` enforces LF line endings.
* [x] **RF6: Access Requests Exhaustive:** 5 Tier-1 datasets, 10 Tier-2 datasets, and 4 external API services documented in `access_requests.md` with official URLs and instructions.
* [x] **RF7: Storage Constraints:** Project installed on partition `D:\` with >180 GB free; `.wslconfig` allocates swap to `D:\wsl\swap.vhdx`.
* [x] **RF8: Kill-and-Resume State Precision:** RNG states (Python, NumPy, PyTorch CPU, CUDA) restored from atomic checkpoint, reproducing identical loss at step 300.
* [x] **RF9: Real Hardware Execution:** Peak allocated memory readings gathered directly from `torch.cuda.max_memory_allocated()` and `nvidia-smi` telemetry.

---

## 4. Plausibility & Literature Comparison

| Metric / Check | Measured Baseline | Literature / Theoretical Value | Auditor Comment |
| :--- | :--- | :--- | :--- |
| **Wilcoxon Min $p$ ($K=5$)** | $p = 0.0625$ | $2^{-4} = 0.0625$ | Exactly matches statistical theory. Paper claim refuted. |
| **Wilcoxon Min $p$ ($K=10$)** | $p = 0.00195$ | $2 \times 2^{-10} = 0.001953$ | Confirms $K \ge 10$ seeds required for $p < 0.01$. |
| **DeBERTa-base VRAM (len 128, b=8)** | 2,168 MiB | ~2.0–2.5 GB (standard HuggingFace benchmark) | Consistent with published Ada Lovelace profiles. |
| **DeBERTa-base VRAM (len 512, b=32)** | 8,032 MiB | ~7.8–8.2 GB | Near physical limit (98.1%); confirms gradient accumulation required for production runs. |

---

## 5. Audit Findings & Severity

| ID | Severity | Area | Finding | Resolution / Verification |
| :--- | :--- | :--- | :--- | :--- |
| **A00-1** | MAJOR (Resolved) | `paths.py` | `_DEFAULT_ROOT` resolved 4 parents up instead of 5, causing `MILESTONES_ROOT.exists()` to check `ares_v2/milestones`. | Resolved: Changed to `parents[4]` (`d:\ares`). Verified via `test_paths_defined` passing. |
| **A00-2** | MAJOR (Resolved) | `results.py` | Bare `except ImportError: pass` triggered silent fallback AST check. | Resolved: Explicitly assigned `"not_installed"` strings on `ImportError`. Verified via `test_no_silent_fallback_ast` passing. |
| **A00-3** | MAJOR (Resolved) | `smoke_kill_resume.py` | `create_synthetic_data` called after `ckpt_mgr.load_latest` reset RNG states, causing $\Delta\text{loss} = 1.96\times 10^{-2}$. | Resolved: Moved data loader creation prior to `load_latest`. Discrepancy dropped to bit-exact $0.00\times 10^0$. |
| **A00-4** | MINOR (Resolved) | `ledger.py` | Loading empty `measured_value` column defaulted to `float64`, crashing on string assignment. | Resolved: Enforced `dtype=str` in `pd.read_csv`. Delta applied cleanly. |

---

## 6. Auditor Verdict

### **VERDICT: CLOSE**

All BLOCKER and MAJOR findings have been fully resolved with regression tests passing. All 9 exit criteria for Milestone M00 are satisfied with mathematical provenance and hardware verification. Milestone M00 is approved for closure and git tagging `m00-complete`.
