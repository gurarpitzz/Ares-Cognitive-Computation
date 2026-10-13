# M13 — Systems Performance: Latency, Throughput, Complexity, Preprocessing Model

**Objective:** Measure every systems claim (Sections III-D, IV-C/E, XI-E, XV, XX, XXVIII-C) on **declared, fixed hardware** with sound timing methodology. Test M01's corrected complexity prediction, and fit the preprocessing-time model from data instead of "demonstration parameters".
**Ledger IDs:** C002, C004, C005, C006, C016, C017, C028, C029, C033, C035, C036, C041, C042, C062 (T_auto), C075 (empirical), C087, C093, C094, C095.
**Depends on:** M11 (frozen models).
**Compute:** The RTX 4060 Laptop (8 GB), i7-13620H, 16 GB DDR5, WSL2: about 10 GPU-h. **This is the hardware every latency and throughput number in the rebuilt paper refers to.** The manuscript's A100 / Xeon / 64 GB statement is REVISED to this hardware (C033, C034). The CPU-only reference runs on the same laptop with `CUDA_VISIBLE_DEVICES=""`.
**Output dir:** `milestones/M13_systems_perf/`

---

## Pre-registration
- **Timing protocol:**
  - Warm-up 50 iterations; ≥ 1000 timed iterations (≥ 200 for video).
  - `torch.cuda.synchronize()` + CUDA events on GPU; `time.perf_counter_ns` on CPU.
  - Model load excluded and reported separately.
  - Report p50 / p90 / p99 / mean with bootstrap CIs.
  - Batch size 1 unless stated; fp32 and fp16/bf16 both reported; SDPA/flash-attention setting recorded.
- **Thermal and power protocol (laptop):**
  - AC power; OMEN Performance mode; fixed fan profile.
  - Background apps closed; Windows Update paused; display on.
  - Start each suite only after the GPU has idled below 50 °C.
  - Log GPU telemetry at 1 s during benchmarks (not 30 s).
  - Run each suite **3 times on different occasions** and report the between-run spread.
  - A suite with thermal throttling detected is re-run. If throttling persists, it is reported as "sustained (throttled)" next to a "burst" number.
  - Record the Windows power plan, NVIDIA driver version and WSL kernel version.
- **End-to-end scopes (C002):** (a) text-only verifier; (b) T2 pipeline with local frozen index (retrieval + S + N for M passages + W + decision); (c) T2 with web retrieval (network-bound; reported separately, never mixed into (b)); (d) per-modality pipelines (image, audio, video).
- **H13.1 (complexity, from M01):** for n ≤ 512 on BERT/RoBERTa/DeBERTa-base, the FLOP log-log slope is close to 1, and wall-clock slope ≤ FLOP slope. A quadratic regime appears only as n approaches ~6d (needs a long-context model).

## Steps

### M13.1 — Benchmark harness
- **Action:** `src/ares/systems/bench.py`: a generic timing harness implementing the protocol; writes results JSON with hardware/software fields; `scripts/m13/run_bench.py --suite <name>`.

### M13.2 — Per-module latency (C016, C017)
- **Action:** Text verifier (DeBERTa-v3-base, n = 128 typical), NLI per passage (and per claim with M passages), retriever (BM25, dense, hybrid), ranker, logic layer, deepfake per video (frames at 1–2 FPS incl. face detection), audio spoof per utterance (4 s), ECAPA per utterance, geo (NER + geotag + snapshot replay), GDELT lookup (replay), fusion head.
- **Output:** `tables/module_latency.csv`, `figures/latency_by_modality.pdf` (replaces Fig. 20 / Tab. XIII).

### M13.3 — End-to-end latency (C002, C028, C062)
- **Action:** Scopes (a)–(d). Detection-to-alert latency in the service pipeline (ingest → verdict → alert queued; uses M14's service, so run after M14.3 or re-run then). T_auto feeds M05's E[T] model (recompute the sensitivity plot).

### M13.4 — Token-length scaling (C041, C042, C075, C087)
- **Action:** BERT-base, RoBERTa-base and DeBERTa-v3-base at n ∈ {16, 32, 64, 128, 256, 384, 512}. The maximum is 512 because RoBERTa can't take more, which documents C042. Also run a long-context encoder (e.g. ModernBERT-base, 8k context, or Longformer-base) at n up to 8192, batch 1; record VRAM and mark any OOM point.
  - For each model: wall-clock (fp16 + fp32) and FLOPs (`torch.utils.flop_counter`).
  - Fit log T = α log n + β over (i) n ≤ 512 and (ii) the long-context range, with bootstrap CIs over repeated timings.
  - Compare with M01's FLOP formula (24nd² + 4n²d) and the predicted crossover n* ≈ 6d.
- **Output:** `figures/latency_vs_tokens.pdf`, `figures/loglog_fit.pdf` (replace Figs. 28/29/31), `tables/scaling_fits.csv`. The manuscript's Table XXII is replaced, not "matched".

### M13.5 — Throughput & concurrency (C004, C029, C036, C093)
- **Action:** Deploy the M14 FastAPI service on the laptop, with dynamic batching (a simple async batcher, or Triton/TorchServe; document which).
  - **Load generator isolation:** run locust/k6 either (i) from a second machine on the wired LAN (preferred; record it), or (ii) on the same laptop, with the client pinned to the E-cores (`taskset -c 12-15`) and the server to the P-cores. Report which was used, because (ii) under-states the server's capacity.
  - Increase concurrency {1, 10, 50, 100, 250, 500, 1000, …} **until saturation** (p99 > 2× the SLO, or errors > 1 %) for scopes (a) and (b). Report req/s and latency percentiles per level, plus the saturation point.
  - Claims/hour = measured sustained throughput × 3600 (C036).
  - Posts/day is an explicit extrapolation, labelled as such (C029).
  - The manuscript's "4000 req/s" (C004) is replaced by the measured saturation throughput on this laptop.
- **Output:** `figures/throughput_vs_concurrency.pdf` (replaces Fig. 3).

### M13.6 — Parallel vs sequential pipeline (C035)
- **Action:** Same inputs through a sequential and a parallel (asyncio/thread-pool/multi-stream) module orchestration; report the latency reduction with CI.

### M13.7 — Preprocessing time model (C005, C006, C094)
- **Action:** Measure T_prep for each modality over a size sweep (text 1–100 kB; image 50 kB–5 MB; audio 0.5–20 MB; video 1–100 MB), ≥ 30 sizes × 20 reps. Fit T = α·S^β + δ by nonlinear least squares (scipy `curve_fit`) with bootstrap CIs; report goodness of fit. Check measured latencies against Table II budgets and report pass/fail per modality (budgets are design targets).
- **Output:** `figures/prep_time_fits.pdf` (replaces Fig. 6), `tables/prep_fit_params.csv`.

### M13.8 — Hardware requirements (C033)
- **Action:** Peak RAM and VRAM per scope (`torch.cuda.max_memory_allocated`, `psutil`). Demonstrate the minimum viable configuration (e.g. CPU-only text pipeline latency on the i7-13620H). Replace the manuscript's "min 32 GB RAM; RTX 3080/A100" (C033) with the measured requirement: it all runs on 8 GB VRAM / 16 GB RAM, with the module-wise footprint stated.

### M13.9 — ONNX / FP16 (C095)
- **Action (optional):** Export the text verifier + NLI to ONNX; measure speedup vs PyTorch at fp16. If not done, ledger → REMOVE the "underway" claim or rewrite it as future work.

### M13.10 — Baseline latencies (Tabs. XXI / XXXVII latency columns)
- **Action:** Measure only baselines we actually run locally (TF-IDF, BERT/RoBERTa/DeBERTa, FakeFlow if run) under identical conditions. Third-party systems not run locally get no latency entry.

---

## Audit gate — milestone-specific red flags
- RF1: CUDA synchronisation present in GPU timing (auditor inspects code and reruns one suite).
- RF2: Model loading / first-call compilation excluded from per-request latency; warm vs cold stated.
- RF3: Network latency never mixed into local-pipeline numbers.
- RF4: Hardware in JSON matches the hardware named in tables.
- RF5: Log-log fits report the n range and CI; no extrapolation beyond the measured range.
- RF6: Load tests run long enough to reach steady state (≥ 2 min per level); errors/timeouts counted.
- RF7: The thermal protocol was followed: telemetry exists for every timing suite, throttled runs are flagged, and there are 3 repeat occasions with their spread.
- RF8: The load-generator placement is stated. If it ran on the same machine, CPU pinning is evidenced.

## Exit criteria
- [ ] All suites run with the protocol; figures/tables regenerated.
- [ ] Complexity prediction tested against measurements (H13.1 outcome stated plainly).
- [ ] Ledger IDs in scope updated (many will be REVISED).
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m13-complete`.
