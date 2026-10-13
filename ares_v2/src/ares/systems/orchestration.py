"""
Parallel vs Sequential Pipeline Orchestration Benchmark (Claim C035).

Measures empirical latency reduction from asynchronous concurrent module execution:
  Delta_par = (T_seq - T_par) / T_seq

Sequential orchestration executes modules serially:
  T_seq = T_text + T_retrieval + T_logic + T_geo + T_fusion

Parallel orchestration executes verification branches concurrently via thread/async pool:
  T_par = max(T_text, T_retrieval, T_logic, T_geo) + T_fusion
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ares.systems.bench import compute_bootstrap_ci

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Comparison of sequential vs parallel pipeline latency."""
    num_runs: int
    mean_seq_ms: float
    p50_seq_ms: float
    p90_seq_ms: float
    mean_par_ms: float
    p50_par_ms: float
    p90_par_ms: float
    delta_par_mean: float
    delta_par_p50: float
    delta_par_ci95: Tuple[float, float]
    claimed_reduction: float = 0.40  # 40% claimed in Section XV-D

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_runs": self.num_runs,
            "mean_seq_ms": round(float(self.mean_seq_ms), 3),
            "p50_seq_ms": round(float(self.p50_seq_ms), 3),
            "p90_seq_ms": round(float(self.p90_seq_ms), 3),
            "mean_par_ms": round(float(self.mean_par_ms), 3),
            "p50_par_ms": round(float(self.p50_par_ms), 3),
            "p90_par_ms": round(float(self.p90_par_ms), 3),
            "delta_par_mean": round(float(self.delta_par_mean), 4),
            "delta_par_p50": round(float(self.delta_par_p50), 4),
            "delta_par_ci95": [round(float(x), 4) for x in self.delta_par_ci95],
            "claimed_reduction": self.claimed_reduction,
        }


def _dummy_task_text() -> float:
    # Simulates text verification inference (~15 ms)
    time.sleep(0.015)
    return 0.85


def _dummy_task_retrieval() -> float:
    # Simulates BM25 + NLI passage scoring (~25 ms)
    time.sleep(0.025)
    return 0.78


def _dummy_task_geo() -> float:
    # Simulates toponym NER + gazetteer (~8 ms)
    time.sleep(0.008)
    return 0.90


def _dummy_task_logic() -> float:
    # Simulates symbolic logic checker (~5 ms)
    time.sleep(0.005)
    return 0.95


def _dummy_task_fusion(s1: float, s2: float, s3: float, s4: float) -> float:
    # Simulates late fusion MLP (~2 ms)
    time.sleep(0.002)
    return float(np.mean([s1, s2, s3, s4]))


def benchmark_orchestration(
    num_runs: int = 50,
    max_workers: int = 4,
    seed: int = 42,
) -> OrchestrationResult:
    """Benchmark sequential vs parallel pipeline execution across repetitions."""
    seq_latencies: List[float] = []
    par_latencies: List[float] = []
    delta_pars: List[float] = []

    # Warmup
    for _ in range(5):
        _ = _dummy_task_text()
        _ = _dummy_task_retrieval()
        _ = _dummy_task_geo()
        _ = _dummy_task_logic()

    logger.info(f"Running {num_runs} orchestration benchmark iterations (Sequential vs Parallel)...")

    for _ in range(num_runs):
        # 1. Sequential run
        t0 = time.perf_counter_ns()
        s1 = _dummy_task_text()
        s2 = _dummy_task_retrieval()
        s3 = _dummy_task_geo()
        s4 = _dummy_task_logic()
        _ = _dummy_task_fusion(s1, s2, s3, s4)
        t1 = time.perf_counter_ns()
        t_seq = (t1 - t0) / 1e6
        seq_latencies.append(t_seq)

        # 2. Parallel run
        t0 = time.perf_counter_ns()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            f1 = executor.submit(_dummy_task_text)
            f2 = executor.submit(_dummy_task_retrieval)
            f3 = executor.submit(_dummy_task_geo)
            f4 = executor.submit(_dummy_task_logic)
            s1_p = f1.result()
            s2_p = f2.result()
            s3_p = f3.result()
            s4_p = f4.result()
        _ = _dummy_task_fusion(s1_p, s2_p, s3_p, s4_p)
        t1 = time.perf_counter_ns()
        t_par = (t1 - t0) / 1e6
        par_latencies.append(t_par)

        delta = (t_seq - t_par) / t_seq
        delta_pars.append(delta)

    seq_arr = np.array(seq_latencies)
    par_arr = np.array(par_latencies)
    delta_arr = np.array(delta_pars)

    mean_seq = float(np.mean(seq_arr))
    p50_seq = float(np.percentile(seq_arr, 50))
    p90_seq = float(np.percentile(seq_arr, 90))

    mean_par = float(np.mean(par_arr))
    p50_par = float(np.percentile(par_arr, 50))
    p90_par = float(np.percentile(par_arr, 90))

    mean_delta = float(np.mean(delta_arr))
    p50_delta = float(np.percentile(delta_arr, 50))
    ci_low, ci_high = compute_bootstrap_ci(delta_arr, num_bootstraps=1000, seed=seed)

    return OrchestrationResult(
        num_runs=num_runs,
        mean_seq_ms=mean_seq,
        p50_seq_ms=p50_seq,
        p90_seq_ms=p90_seq,
        mean_par_ms=mean_par,
        p50_par_ms=p50_par,
        p90_par_ms=p90_par,
        delta_par_mean=mean_delta,
        delta_par_p50=p50_delta,
        delta_par_ci95=(ci_low, ci_high),
        claimed_reduction=0.40,
    )
