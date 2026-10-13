"""
Throughput, Concurrency & Saturation Benchmarking (Claims C004, C029, C036, C093).

Rigorous empirical load testing of the ARES verification service:
1. Concurrency sweep: C in {1, 5, 10, 25, 50, 100, 250, 500}.
2. Async worker pool querying FastAPI service via ASGITransport (eliminates external network noise, RF3).
3. Steady-state saturation detection (RF6):
   Saturated when p99 latency > 1000 ms or error rate > 1.0%.
4. Capacity calculations:
   - Sustained throughput (req/s)
   - Claims per hour = req/s * 3600 (Claim C036)
   - Posts per day = req/s * 86400 (Claim C029)
5. Replaces manuscript ungrounded claim of '~4000 req/s under 5000 concurrency' (Claim C004).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from ares.service.app import app
from ares.systems.bench import compute_bootstrap_ci

logger = logging.getLogger(__name__)

# Suppress per-request logging from httpx and httpcore during high-concurrency sweeps
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class ConcurrencyResult:
    """Performance metrics for a single concurrency level."""
    concurrency: int
    duration_s: float
    total_requests: int
    throughput_req_per_s: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    ci95_low_ms: float
    ci95_high_ms: float
    error_count: int
    error_rate: float
    is_saturated: bool
    claims_per_hour: float
    posts_per_day: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "duration_s": round(float(self.duration_s), 2),
            "total_requests": self.total_requests,
            "throughput_req_per_s": round(float(self.throughput_req_per_s), 2),
            "p50_ms": round(float(self.p50_ms), 3),
            "p90_ms": round(float(self.p90_ms), 3),
            "p99_ms": round(float(self.p99_ms), 3),
            "mean_ms": round(float(self.mean_ms), 3),
            "std_ms": round(float(self.std_ms), 3),
            "ci95_low_ms": round(float(self.ci95_low_ms), 3),
            "ci95_high_ms": round(float(self.ci95_high_ms), 3),
            "error_count": self.error_count,
            "error_rate": round(float(self.error_rate), 4),
            "is_saturated": self.is_saturated,
            "claims_per_hour": round(float(self.claims_per_hour), 1),
            "posts_per_day": round(float(self.posts_per_day), 1),
        }


async def _run_worker(
    client: httpx.AsyncClient,
    end_time: float,
    latencies: List[float],
    errors: List[str],
) -> None:
    """Async worker continuously querying /verify until end_time."""
    req_body = {
        "claim_id": "bench-claim-001",
        "claim_text": "Government declares emergency disaster flood relief for three regional districts.",
        "has_media": False,
    }

    while time.perf_counter() < end_time:
        t0 = time.perf_counter_ns()
        try:
            resp = await client.post("/verify", json=req_body, timeout=5.0)
            t1 = time.perf_counter_ns()
            if resp.status_code == 200:
                latencies.append((t1 - t0) / 1e6)
            else:
                errors.append(f"Status {resp.status_code}")
        except Exception as e:
            errors.append(str(e))
        # Cooperatively yield to event loop to allow concurrent workers to execute
        await asyncio.sleep(0)


async def benchmark_concurrency_step(
    concurrency: int,
    duration_s: float = 3.0,
    seed: int = 42,
) -> ConcurrencyResult:
    """Run load test at a specific concurrency level."""
    latencies: List[float] = []
    errors: List[str] = []

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        t_start = time.perf_counter()
        end_time = t_start + duration_s
        workers = [
            asyncio.create_task(_run_worker(client, end_time, latencies, errors))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers, return_exceptions=True)
        t_end = time.perf_counter()

    actual_duration = t_end - t_start
    total_reqs = len(latencies)
    err_count = len(errors)
    total_attempts = total_reqs + err_count
    err_rate = err_count / total_attempts if total_attempts > 0 else 0.0

    throughput = total_reqs / actual_duration if actual_duration > 0 else 0.0

    if latencies:
        arr = np.array(latencies, dtype=np.float64)
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p99 = float(np.percentile(arr, 99))
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        # Subsample for bootstrap CI if sample size is very large
        if len(arr) > 5000:
            rng = np.random.default_rng(seed)
            sample_ci = rng.choice(arr, size=5000, replace=False)
        else:
            sample_ci = arr
        ci_low, ci_high = compute_bootstrap_ci(sample_ci, num_bootstraps=1000, seed=seed)
    else:
        p50 = p90 = p99 = mean_val = std_val = ci_low = ci_high = 0.0

    # Saturation rule (RF6): p99 > 1000 ms or error rate > 1.0%
    is_saturated = p99 > 1000.0 or err_rate > 0.01

    claims_per_hour = throughput * 3600.0
    posts_per_day = throughput * 86400.0

    return ConcurrencyResult(
        concurrency=concurrency,
        duration_s=actual_duration,
        total_requests=total_reqs,
        throughput_req_per_s=throughput,
        p50_ms=p50,
        p90_ms=p90,
        p99_ms=p99,
        mean_ms=mean_val,
        std_ms=std_val,
        ci95_low_ms=ci_low,
        ci95_high_ms=ci_high,
        error_count=err_count,
        error_rate=err_rate,
        is_saturated=is_saturated,
        claims_per_hour=claims_per_hour,
        posts_per_day=posts_per_day,
    )


def run_concurrency_grid(
    concurrency_grid: Optional[List[int]] = None,
    duration_per_step_s: float = 5.0,
    seed: int = 42,
) -> List[ConcurrencyResult]:
    """Execute complete concurrency grid load benchmark."""
    if concurrency_grid is None:
        concurrency_grid = [1, 5, 10, 25, 50, 100, 250, 500]

    results: List[ConcurrencyResult] = []

    for c in concurrency_grid:
        logger.info(f"Running concurrency level C = {c} for {duration_per_step_s}s...")
        res = asyncio.run(benchmark_concurrency_step(c, duration_s=duration_per_step_s, seed=seed))
        results.append(res)
        logger.info(
            f"C={c}: Throughput={res.throughput_req_per_s:.1f} req/s, p50={res.p50_ms:.2f}ms, "
            f"p99={res.p99_ms:.2f}ms, Errors={res.error_count} ({res.error_rate*100:.2f}%), "
            f"Saturated={'YES' if res.is_saturated else 'NO'}"
        )

    return results
