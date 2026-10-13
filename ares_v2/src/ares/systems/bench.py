"""
High-Resolution Systems Benchmarking Harness for ARES Rebuild.

Implements rigorous hardware timing protocols (Audit Red Flags RF1, RF2, RF3, RF4, RF7):
- GPU timing via torch.cuda.Event with explicit torch.cuda.synchronize() boundaries.
- CPU timing via high-resolution monotonic clocks (time.perf_counter_ns).
- Strict separation of cold-start initialization from per-inference latency distributions.
- Non-parametric percentile bootstrap confidence intervals (B = 1000).
- Real-time hardware telemetry (GPU temperature, VRAM allocation, CPU RAM).
"""

from __future__ import annotations

import gc
import logging
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class HardwareTelemetry:
    """Snapshot of hardware telemetry at time of measurement."""
    device_name: str
    gpu_temperature_c: Optional[float]
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    gpu_max_memory_allocated_mb: float
    cpu_ram_used_gb: float
    cpu_ram_total_gb: float
    cpu_percent: float
    driver_version: Optional[str]
    cuda_version: str
    torch_version: str
    python_version: str
    os_version: str


@dataclass
class TimingDistribution:
    """Latency distribution with robust summary statistics and bootstrap CIs."""
    name: str
    device: str
    dtype: str
    num_warmup: int
    num_timed: int
    cold_start_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    ci95_low_ms: float
    ci95_high_ms: float
    min_ms: float
    max_ms: float
    telemetry: Optional[HardwareTelemetry] = None
    latencies_ms: List[float] = field(default_factory=list)

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "name": self.name,
            "device": self.device,
            "dtype": self.dtype,
            "num_warmup": self.num_warmup,
            "num_timed": self.num_timed,
            "cold_start_ms": round(float(self.cold_start_ms), 4),
            "p50_ms": round(float(self.p50_ms), 4),
            "p90_ms": round(float(self.p90_ms), 4),
            "p99_ms": round(float(self.p99_ms), 4),
            "mean_ms": round(float(self.mean_ms), 4),
            "std_ms": round(float(self.std_ms), 4),
            "ci95_low_ms": round(float(self.ci95_low_ms), 4),
            "ci95_high_ms": round(float(self.ci95_high_ms), 4),
            "min_ms": round(float(self.min_ms), 4),
            "max_ms": round(float(self.max_ms), 4),
        }
        if self.telemetry:
            d["telemetry"] = {
                "device_name": self.telemetry.device_name,
                "gpu_temperature_c": self.telemetry.gpu_temperature_c,
                "gpu_memory_allocated_mb": round(self.telemetry.gpu_memory_allocated_mb, 2),
                "gpu_max_memory_allocated_mb": round(self.telemetry.gpu_max_memory_allocated_mb, 2),
                "cpu_ram_used_gb": round(self.telemetry.cpu_ram_used_gb, 2),
                "cpu_ram_total_gb": round(self.telemetry.cpu_ram_total_gb, 2),
                "driver_version": self.telemetry.driver_version,
            }
        if include_raw:
            d["latencies_ms"] = [round(float(x), 4) for x in self.latencies_ms]
        return d


def get_hardware_telemetry(device: str = "cuda") -> HardwareTelemetry:
    """Capture current hardware state, temperatures, and memory utilization."""
    device_name = "CPU"
    gpu_temp = None
    vram_alloc = 0.0
    vram_res = 0.0
    vram_max = 0.0
    driver_version = None

    if device == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_alloc = torch.cuda.memory_allocated(0) / (1024 * 1024)
        vram_res = torch.cuda.memory_reserved(0) / (1024 * 1024)
        vram_max = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu,driver_version", "--format=csv,noheader,nounits"],
                text=True,
                timeout=2,
            )
            parts = [p.strip() for p in out.strip().split(",")]
            if len(parts) >= 1 and parts[0].isdigit():
                gpu_temp = float(parts[0])
            if len(parts) >= 2:
                driver_version = parts[1]
        except Exception as e:
            logger.debug(f"Telemetry query via nvidia-smi failed: {e}")

    vm = psutil.virtual_memory()
    ram_used_gb = (vm.total - vm.available) / (1024 ** 3)
    ram_total_gb = vm.total / (1024 ** 3)
    cpu_percent = psutil.cpu_percent(interval=None)

    return HardwareTelemetry(
        device_name=device_name,
        gpu_temperature_c=gpu_temp,
        gpu_memory_allocated_mb=vram_alloc,
        gpu_memory_reserved_mb=vram_res,
        gpu_max_memory_allocated_mb=vram_max,
        cpu_ram_used_gb=ram_used_gb,
        cpu_ram_total_gb=ram_total_gb,
        cpu_percent=cpu_percent,
        driver_version=driver_version,
        cuda_version=torch.version.cuda or "N/A",
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        os_version=f"{platform.system()} {platform.release()}",
    )


def compute_bootstrap_ci(
    data: np.ndarray,
    num_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute non-parametric percentile bootstrap confidence interval on mean."""
    if len(data) == 0:
        return 0.0, 0.0
    if len(data) == 1:
        return float(data[0]), float(data[0])

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, len(data), size=(num_bootstraps, len(data)))
    boot_means = np.mean(data[boot_indices], axis=1)

    alpha = 1.0 - ci
    low_percentile = (alpha / 2.0) * 100.0
    high_percentile = (1.0 - alpha / 2.0) * 100.0

    ci_low = float(np.percentile(boot_means, low_percentile))
    ci_high = float(np.percentile(boot_means, high_percentile))
    return ci_low, ci_high


class BenchmarkHarness:
    """Standardized timing harness enforcing rigorous systems benchmarking."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run_benchmark(
        self,
        name: str,
        fn: Callable[[], Any],
        num_warmup: int = 50,
        num_timed: int = 1000,
        device: str = "cuda",
        dtype: str = "fp32",
        setup_fn: Optional[Callable[[], None]] = None,
    ) -> TimingDistribution:
        """Execute benchmark with cold-start isolation, warm-up, and timing boundaries.
        
        Args:
            name: Identifier for benchmark suite/component.
            fn: Zero-argument callable to time.
            num_warmup: Number of warm-up iterations (excluded from statistics).
            num_timed: Number of timed iterations to collect.
            device: 'cuda' or 'cpu'.
            dtype: 'fp32', 'fp16', or 'bf16'.
            setup_fn: Optional callable run before starting.
        """
        use_cuda = device == "cuda" and torch.cuda.is_available()

        if setup_fn is not None:
            setup_fn()

        # Reset peak memory stats before starting
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()

        # 1. Cold-start measurement (Strictly isolated, RF2)
        cold_start_ms = 0.0
        if use_cuda:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()
            cold_start_ms = start_event.elapsed_time(end_event)
        else:
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            cold_start_ms = (t1 - t0) / 1e6

        # 2. Warm-up iterations (Settles kernel launches, JIT, caching)
        for _ in range(num_warmup):
            fn()
        if use_cuda:
            torch.cuda.synchronize()

        # 3. Timed execution loop (Strict RF1 synchronization)
        latencies_ms: List[float] = []

        if use_cuda:
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_timed)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_timed)]

            for i in range(num_timed):
                torch.cuda.synchronize()
                start_events[i].record()
                fn()
                end_events[i].record()

            torch.cuda.synchronize()
            latencies_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(num_timed)]
        else:
            for _ in range(num_timed):
                t0 = time.perf_counter_ns()
                fn()
                t1 = time.perf_counter_ns()
                latencies_ms.append((t1 - t0) / 1e6)

        arr = np.array(latencies_ms, dtype=np.float64)
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p99 = float(np.percentile(arr, 99))
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci_low, ci_high = compute_bootstrap_ci(arr, num_bootstraps=1000, seed=self.seed)

        telemetry = get_hardware_telemetry(device=device if use_cuda else "cpu")

        return TimingDistribution(
            name=name,
            device="cuda" if use_cuda else "cpu",
            dtype=dtype,
            num_warmup=num_warmup,
            num_timed=num_timed,
            cold_start_ms=cold_start_ms,
            p50_ms=p50,
            p90_ms=p90,
            p99_ms=p99,
            mean_ms=mean_val,
            std_ms=std_val,
            ci95_low_ms=ci_low,
            ci95_high_ms=ci_high,
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            telemetry=telemetry,
            latencies_ms=latencies_ms,
        )
