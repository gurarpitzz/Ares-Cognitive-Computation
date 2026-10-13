"""
Unit and Integration Tests for Systems Performance & Complexity (Milestone M13).
"""

import time
import numpy as np
import pytest
import torch

from ares.systems.bench import BenchmarkHarness, compute_bootstrap_ci, get_hardware_telemetry
from ares.systems.scaling import (
    compute_theoretical_flop_slope,
    compute_theoretical_transformer_flops,
    fit_power_law,
    verify_roberta_architectural_limit,
)
from ares.systems.preprocessing import power_law_func, fit_preprocessing_curve
from ares.systems.orchestration import benchmark_orchestration
from ares.systems.throughput import ConcurrencyResult


def test_bootstrap_ci_coverage():
    """Verify bootstrap confidence interval bounds mean correctly."""
    data = np.array([10.0, 11.0, 12.0, 10.5, 11.5, 9.5, 12.5], dtype=np.float64)
    low, high = compute_bootstrap_ci(data, num_bootstraps=500, ci=0.95, seed=42)
    mean_val = float(np.mean(data))
    assert low <= mean_val <= high
    assert low < high


def test_benchmark_harness_cold_start_and_warmup():
    """Verify BenchmarkHarness isolates cold start and executes declared warmup/timed counts."""
    call_counts = {"count": 0}

    def dummy_op():
        call_counts["count"] += 1
        time.sleep(0.001)

    harness = BenchmarkHarness(seed=42)
    dist = harness.run_benchmark(
        name="test_dummy",
        fn=dummy_op,
        num_warmup=5,
        num_timed=10,
        device="cpu",
    )

    # Total calls = 1 (cold start) + 5 (warmup) + 10 (timed) = 16
    assert call_counts["count"] == 16
    assert dist.num_warmup == 5
    assert dist.num_timed == 10
    assert dist.cold_start_ms > 0.0
    assert dist.p50_ms > 0.0
    assert dist.ci95_low_ms <= dist.mean_ms <= dist.ci95_high_ms


def test_transformer_theoretical_flops_and_slope():
    """Verify analytical transformer FLOP formulas and slope bound alpha < 1.4 for n <= 512."""
    flops_128 = compute_theoretical_transformer_flops(128, d=768, num_layers=12)
    flops_512 = compute_theoretical_transformer_flops(512, d=768, num_layers=12)

    assert flops_512 > flops_128 > 0

    slope_128 = compute_theoretical_flop_slope(128, d=768)
    slope_512 = compute_theoretical_flop_slope(512, d=768)

    # FLOP slope must be close to 1.0 (sub-quadratic) because linear term dominates
    assert 1.0 <= slope_128 <= 1.05
    assert 1.05 <= slope_512 <= 1.15
    assert slope_512 < 1.4  # Strictly refutes alpha = 1.99


def test_fit_power_law_synthetic_recovery():
    """Verify OLS power-law fitting recovers true synthetic parameters."""
    seq_lens = [16, 32, 64, 128, 256, 512]
    true_alpha = 1.20
    true_beta = -2.50

    # T = exp(beta) * n^alpha
    latencies = [float(np.exp(true_beta) * (n ** true_alpha)) for n in seq_lens]

    fit = fit_power_law(seq_lens, latencies, num_bootstraps=200, seed=42)
    assert abs(fit.alpha - true_alpha) < 0.01
    assert abs(fit.beta - true_beta) < 0.01
    assert fit.r_squared > 0.999
    assert fit.alpha_ci95_low <= fit.alpha <= fit.alpha_ci95_high


def test_roberta_architectural_limit_detection():
    """Verify RoBERTa position limit check correctly identifies n > 512 constraint."""
    # Test on CPU to ensure quick deterministic unit test
    results = verify_roberta_architectural_limit(device="cpu", test_lengths=[256, 768])

    assert 256 in results
    assert 768 in results

    assert results[256]["status"] == "SUCCESS"
    assert results[256]["exceeds_limit"] is False

    assert results[768]["status"] == "CRASHED_EXPECTED"
    assert results[768]["exceeds_limit"] is True
    assert "IndexError" in results[768]["error_type"] or "RuntimeError" in results[768]["error_type"]


def test_fit_preprocessing_curve_synthetic():
    """Verify power-law curve fitting on synthetic preprocessing data."""
    sizes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    true_a, true_b, true_d = 2.0, 0.8, 0.5
    means = [power_law_func(s, true_a, true_b, true_d) for s in sizes]
    stds = [0.05 * m for m in means]

    fit = fit_preprocessing_curve(
        modality="test_mod",
        sizes_mb=sizes,
        means_ms=means,
        stds_ms=stds,
        typical_size_mb=1.0,
        target_budget_ms=10.0,
        num_bootstraps=50,
        seed=42,
    )

    assert fit.r_squared > 0.98
    assert abs(fit.beta - true_b) < 0.15
    assert fit.budget_met is True


def test_orchestration_latency_reduction():
    """Verify parallel orchestration achieves non-zero latency reduction."""
    res = benchmark_orchestration(num_runs=5, max_workers=4, seed=42)
    assert res.num_runs == 5
    assert res.mean_seq_ms > res.mean_par_ms
    assert res.delta_par_mean > 0.0
    assert 0.0 < res.delta_par_p50 < 1.0


def test_concurrency_result_serialization():
    """Verify ConcurrencyResult dataclass serialization."""
    cr = ConcurrencyResult(
        concurrency=10,
        duration_s=5.0,
        total_requests=500,
        throughput_req_per_s=100.0,
        p50_ms=8.5,
        p90_ms=12.0,
        p99_ms=18.5,
        mean_ms=9.1,
        std_ms=2.1,
        ci95_low_ms=8.9,
        ci95_high_ms=9.3,
        error_count=0,
        error_rate=0.0,
        is_saturated=False,
        claims_per_hour=360000.0,
        posts_per_day=8640000.0,
    )
    d = cr.to_dict()
    assert d["concurrency"] == 10
    assert d["throughput_req_per_s"] == 100.0
    assert d["claims_per_hour"] == 360000.0
    assert d["is_saturated"] is False
