"""ARES systems performance, benchmarking, complexity, and scaling module."""

from ares.systems.bench import (
    BenchmarkHarness,
    HardwareTelemetry,
    TimingDistribution,
    compute_bootstrap_ci,
    get_hardware_telemetry,
)
from ares.systems.footprint import FootprintProfile, audit_hardware_footprint
from ares.systems.modules import (
    benchmark_all_modules,
    benchmark_audio_spoofing,
    benchmark_bm25_retrieval,
    benchmark_late_fusion_head,
    benchmark_logic_checker,
    benchmark_nli_passage_scoring,
    benchmark_text_verifier,
    benchmark_toponym_geo,
    benchmark_video_deepfake,
)
from ares.systems.orchestration import OrchestrationResult, benchmark_orchestration
from ares.systems.preprocessing import (
    PreprocessingModelFit,
    fit_preprocessing_curve,
    sweep_all_preprocessing,
)
from ares.systems.scaling import (
    PowerLawFit,
    compute_theoretical_flop_slope,
    compute_theoretical_transformer_flops,
    fit_power_law,
    sweep_long_context_scaling,
    sweep_token_length_scaling,
    verify_roberta_architectural_limit,
)
from ares.systems.throughput import (
    ConcurrencyResult,
    benchmark_concurrency_step,
    run_concurrency_grid,
)

__all__ = [
    "BenchmarkHarness",
    "HardwareTelemetry",
    "TimingDistribution",
    "compute_bootstrap_ci",
    "get_hardware_telemetry",
    "benchmark_all_modules",
    "benchmark_text_verifier",
    "benchmark_nli_passage_scoring",
    "benchmark_bm25_retrieval",
    "benchmark_logic_checker",
    "benchmark_video_deepfake",
    "benchmark_audio_spoofing",
    "benchmark_toponym_geo",
    "benchmark_late_fusion_head",
    "PowerLawFit",
    "compute_theoretical_flop_slope",
    "compute_theoretical_transformer_flops",
    "fit_power_law",
    "sweep_token_length_scaling",
    "sweep_long_context_scaling",
    "verify_roberta_architectural_limit",
    "PreprocessingModelFit",
    "fit_preprocessing_curve",
    "sweep_all_preprocessing",
    "ConcurrencyResult",
    "benchmark_concurrency_step",
    "run_concurrency_grid",
    "OrchestrationResult",
    "benchmark_orchestration",
    "FootprintProfile",
    "audit_hardware_footprint",
]
