# Pre-Registration: Milestone M13 — Systems Performance, Latency, Throughput & Complexity Benchmarks

**Milestone:** M13 — Systems Performance: Latency, Throughput, Complexity, Preprocessing Model  
**Repository:** `gurarpitzz/Ares-Cognitive-Computation`  
**Date:** October 14, 2026  
**Status:** PRE-REGISTERED  
**Commit Baseline:** `5a5d2a0` (`m12-complete`, `ares-synth-v1`)  
**Hardware Environment:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6, 140W TGP), Intel Core i7-13620H (10 cores, 16 threads), 16 GB DDR5 RAM, 1 TB NVMe SSD  
**Software Environment:** Windows 11 Home 24H2, NVIDIA Driver 572.70, CUDA 12.4, Python 3.11.9, PyTorch 2.6.0+cu124, transformers 4.49.0  
**Target Ledger Claims:** `C002`, `C004`, `C005`, `C006`, `C016`, `C017`, `C028`, `C029`, `C033`, `C035`, `C036`, `C041`, `C042`, `C062`, `C075`, `C087`, `C093`, `C094`, `C095`  

---

## 1. Scientific Objective and Context

Sections III-D, IV-C/E, XI-E, XV, XX, and XXVIII-C of the ARES manuscript make bold systems and efficiency assertions:
1. **Monolithic Latency Claim (`C002`):** Abstract and Section XV-E claim an end-to-end latency of "415 ms per claim" without defining whether this includes web retrieval, evidence cross-scoring, or multimodal verification.
2. **Ungrounded Throughput Claim (`C004`):** Section III-D claims "near-linear throughput under increasing concurrent load up to ~5000 concurrent requests (~4000 req/s)" on hardware that was never specified in code.
3. **Fictitious Preprocessing Model Parameters (`C006`, `C094`):** The power-law preprocessing model $T_{\text{prep}} = \alpha \cdot S^\beta + \delta$ was parameterized using illustrative values selected "for demonstration" rather than fitted from empirical measurements.
4. **Complexity Law Arithmetic Contradiction (Audit Issues I-8, M-4, Claims `C041`, `C075`, `C087`):** The manuscript claims latency scales quadratically with token length ($\alpha = 1.99 \pm 0.04$, Table XXII) because "for typical $n < d$, the quadratic attention term dominates". As proven in `01_MANUSCRIPT_AUDIT.md` §3 and Milestone M01, for $n < d$ (e.g., $n \le 512, d = 768$), $n^2 d < n d^2$, meaning the linear/projection FFN term dominates, producing an empirical slope $\alpha \approx 1.0–1.4$, not $2.0$.
5. **Architectural Limit Violation (Audit Issue M-3, Claim `C042`):** Table XXII claims to measure fine-tuned RoBERTa-base at $n = 768$ and $n = 1024$. Standard RoBERTa-base possesses only 512 learned positional embeddings (`max_position_embeddings = 514`); passing $n \ge 514$ causes an immediate architectural exception.
6. **Hardware Spec Correction (`C033`):** The manuscript claimed a minimum requirement of "32 GB RAM, RTX 3080/A100". In this rebuild, the entire system is profiled on an HP OMEN 16 laptop (RTX 4060 8 GB, 16 GB RAM), establishing the true minimum viable footprint.

Milestone M13 replaces all legacy numbers with rigorous, hardware-grounded timing benchmarks following strict systems measurement protocols.

---

## 2. Rigorous Timing & Benchmarking Protocol (Audit Red Flags RF1, RF2, RF3)

### 2.1 Synchronization & Clock Accuracy
- **GPU Timing (RF1):** Evaluated strictly via `torch.cuda.Event(enable_timing=True)` wrapping the target forward execution, flanked by explicit `torch.cuda.synchronize()` calls before and after event recording.
- **CPU Timing:** Evaluated strictly using high-resolution monotonic clocks via `time.perf_counter_ns()`.
- **Exclusion of Cold Starts (RF2):** Model initialization, checkpoint weight deserialization, CUDA context creation, and initial JIT kernel compilation are timed separately as "Cold-Start Initialization" and strictly excluded from per-inference latency distributions.
- **Warm-Up Protocol:** Minimum 50 warm-up iterations prior to recording; minimum 1000 timed iterations for text/audio/tabular modules, minimum 200 iterations for video processing.
- **Reporting Metrics:** Median ($p_{50}$), $p_{90}$, $p_{99}$, mean, standard deviation, and 95% bootstrap confidence intervals ($B = 1000$).

### 2.2 Thermal & Telemetry Protocol (Audit Red Flag RF7)
- Device connected to continuous AC mains power under HP OMEN "Performance" profile with fixed cooling fan curves.
- GPU baseline temperature verified $< 50^\circ\text{C}$ prior to initiating each benchmark suite.
- Real-time logging of GPU temperature, core frequency, memory utilization, and thermal throttle flags.

---

## 3. Experimental Design & Scope Specifications

### 3.1 End-to-End Latency Scopes (Claim `C002`, `C028`, `C062`)
Latency is evaluated across 4 well-defined architectural scopes:
- **Scope A (Text Verifier Alone):** Tokenization + DeBERTa-v3-base forward pass ($n = 128$) + calibrated LLR scoring.
- **Scope B (Full T2 Verification with Local Corpus):** Claim query parsing + BM25 top-5 passage retrieval over local Wikipedia corpus + MiniLM cross-scoring + BART-NLI contextual inference + logic layer check + LLR fusion.
- **Scope C (Vision Deepfake Pipeline):** Face detection on video frames (1–2 FPS) + ResNet-50 feature extraction + temporal sequence aggregation.
- **Scope D (Audio Tampering Pipeline):** 4-second speech audio waveform $\to$ mel-spectrogram extraction + Declared CNN inference + voice spoof probability.
- **Scope E (Detection-to-Alert Pipeline, Claim `C028`):** Ingestion of claim $\to$ pipeline execution $\to$ threshold check $\to$ alert queuing.

### 3.2 Token-Length Complexity & Transformer Scaling (Claims `C041`, `C042`, `C075`, `C087`)
- **Models:** DeBERTa-v3-base, RoBERTa-base, and BERT-base.
- **Token Sweep:** $n \in \{16, 32, 64, 128, 256, 384, 512\}$.
- **Architectural Boundary Verification (`C042`):** Attempt inference on RoBERTa-base at $n \in \{768, 1024\}$ and document the exact PyTorch exception confirming the impossibility of Table XXII.
- **Long-Context Benchmark:** ModernBERT-base / Longformer evaluated on $n \in \{512, 1024, 2048, 4096\}$.
- **Power-Law Regression:** Fit $\log T(n) = \alpha \log n + \beta$ over $n \le 512$ and evaluate whether $\alpha \approx 1.0–1.4$ (linear/sub-quadratic) holds as predicted by M01 (Hypothesis H13.1).

### 3.3 Throughput & Concurrency Benchmark (Claims `C004`, `C029`, `C036`, `C093`)
- **Server Deployment:** FastAPI service running on the local host with async task pooling.
- **Concurrency Grid:** $C \in \{1, 5, 10, 25, 50, 100, 250, 500\}$ concurrent worker requests.
- **Saturation Rule (RF6):** Steady-state evaluation ($\ge 30$ seconds per load step). System is saturated when $p_{99} > 1000\text{ ms}$ ($2\times$ SLO) or error rate exceeds $1.0\%$.
- **Reported Capacity:** Peak sustained throughput (req/s), hourly throughput ($\text{req/s} \times 3600$, Claim `C036`), and extrapolated daily throughput ($\text{req/s} \times 86400$, Claim `C029`).

### 3.4 Preprocessing Power-Law Model (Claims `C005`, `C006`, `C094`)
- **Model:** $T_{\text{prep}}(S) = \alpha \cdot S^\beta + \delta$
- **Size Grids:**
  - Text: $1\text{ kB}$ to $100\text{ kB}$ (30 size steps $\times 20$ reps)
  - Image: $50\text{ kB}$ to $5\text{ MB}$ (30 size steps $\times 20$ reps)
  - Audio: $0.5\text{ MB}$ to $20\text{ MB}$ (30 size steps $\times 20$ reps)
  - Video: $1\text{ MB}$ to $100\text{ MB}$ (30 size steps $\times 20$ reps)
- **Fitting:** Non-linear least squares optimization (`scipy.optimize.curve_fit`) fitting empirical parameters $\hat{\alpha}, \hat{\beta}, \hat{\delta}$ with $R^2$ goodness-of-fit and 95% bootstrap confidence intervals.
- **SLO Verification (`C005`):** Check measured preprocessing latencies against Table II design targets (text $\le 50\text{ ms}$, image $\le 100\text{ ms}$, audio $\le 150\text{ ms}$, video $\le 250\text{ ms}$).

### 3.5 Parallel vs Sequential Orchestration (Claim `C035`)
- Compare sequential pipeline execution against parallel asynchronous task execution across multimodal modules.
- Measure latency reduction $\Delta_{\text{par}} = \frac{T_{\text{seq}} - T_{\text{par}}}{T_{\text{seq}}}$ with 95% bootstrap confidence intervals.

---

## 4. Audit Gates & Red Flags

- **RF1 (CUDA Synchronization):** Every GPU measurement is bounded by CUDA event records and explicit synchronization.
- **RF2 (Cold-Start Separation):** Model loading and first-call allocations are strictly separated from per-claim inference timings.
- **RF3 (No Network Conflation):** Network retrieval times are never mixed into local computational pipeline benchmarks.
- **RF4 (Hardware Reporting):** Hardware specifications in JSON results match the physical benchmark machine (RTX 4060 Laptop, i7-13620H).
- **RF5 (Log-Log Fit Rigor):** Power-law fits report exact token ranges, confidence intervals, and residuals without post-hoc truncation.
- **RF6 (Steady-State Concurrency):** Load tests run under steady-state conditions; timeouts and connection drops are strictly audited.
- **RF7 (Thermal Integrity):** Device thermal conditions are monitored to ensure absence of throttling during benchmark runs.
