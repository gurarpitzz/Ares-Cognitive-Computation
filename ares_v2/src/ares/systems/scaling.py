"""
Transformer Token-Length Complexity & Architectural Boundary Scaling (Claims C041, C042, C075, C087).

Rigorous empirical analysis of transformer encoder latency scaling:
1. Sweep sequence lengths n in {16, 32, 64, 128, 256, 384, 512}.
2. Fit empirical power-law log T = alpha * log n + beta with 95% bootstrap CIs.
3. Test Hypothesis H13.1 (from M01): FLOP slope alpha_FLOP in [1.0, 1.10] for n <= 512;
   wall-clock slope alpha <= 1.4 << 2.0, directly refuting the manuscript's claimed quadratic scaling slope.
4. Architectural Limit Verification (Claim C042, Audit Issue M-3):
   Demonstrate that standard RoBERTa-base (max_position_embeddings = 514) crashes at n in {768, 1024}.
5. Long-context attention scaling up to n = 4096 demonstrating the true theoretical crossover point n* = 6d.
"""

from __future__ import annotations

import logging
import math
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from ares.systems.bench import BenchmarkHarness, TimingDistribution, compute_bootstrap_ci

logger = logging.getLogger(__name__)


@dataclass
class PowerLawFit:
    """Parameters of fitted power law log T = alpha * log n + beta."""
    alpha: float
    beta: float
    r_squared: float
    alpha_ci95_low: float
    alpha_ci95_high: float
    seq_lens: List[int]
    measured_latencies_ms: List[float]
    predicted_latencies_ms: List[float]


def compute_theoretical_transformer_flops(n: int, d: int = 768, num_layers: int = 12) -> int:
    """Exact FLOPs for 12-layer transformer encoder forward pass.
    
    Per layer:
      - Q, K, V, O projections: 4 * 2 * n * d^2 = 8 * n * d^2
      - Attention score (QK^T) & context (AV): 2 * 2 * n^2 * d = 4 * n^2 * d
      - FFN (two projections with hidden dim 4d): 2 * 2 * 4 * n * d^2 = 16 * n * d^2
      Total per layer = 24 * n * d^2 + 4 * n^2 * d
    """
    per_layer = 24 * n * (d ** 2) + 4 * (n ** 2) * d
    return num_layers * per_layer


def compute_theoretical_flop_slope(n: int, d: int = 768) -> float:
    """Exact analytical derivative d(log FLOPs) / d(log n) = (24d + 8n) / (24d + 4n)."""
    return (24.0 * d + 8.0 * n) / (24.0 * d + 4.0 * n)


def fit_power_law(
    seq_lens: List[int],
    latencies_ms: List[float],
    num_bootstraps: int = 1000,
    seed: int = 42,
) -> PowerLawFit:
    """Fit log T = alpha * log n + beta via Ordinary Least Squares with bootstrap CIs."""
    x = np.log(np.array(seq_lens, dtype=np.float64))
    y = np.log(np.array(latencies_ms, dtype=np.float64))

    # OLS fit
    coeffs = np.polyfit(x, y, deg=1)
    alpha = float(coeffs[0])
    beta = float(coeffs[1])

    y_pred = alpha * x + beta
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    # Percentile bootstrap on slope alpha
    rng = np.random.default_rng(seed)
    n_points = len(x)
    boot_alphas = []
    for _ in range(num_bootstraps):
        idx = rng.integers(0, n_points, size=n_points)
        bx, by = x[idx], y[idx]
        if len(np.unique(bx)) > 1:
            b_coeffs = np.polyfit(bx, by, deg=1)
            boot_alphas.append(b_coeffs[0])
        else:
            boot_alphas.append(alpha)

    boot_arr = np.array(boot_alphas)
    ci_low = float(np.percentile(boot_arr, 2.5))
    ci_high = float(np.percentile(boot_arr, 97.5))

    return PowerLawFit(
        alpha=alpha,
        beta=beta,
        r_squared=r_squared,
        alpha_ci95_low=ci_low,
        alpha_ci95_high=ci_high,
        seq_lens=seq_lens,
        measured_latencies_ms=latencies_ms,
        predicted_latencies_ms=[float(np.exp(yp)) for yp in y_pred],
    )


def sweep_token_length_scaling(
    harness: BenchmarkHarness,
    model_name: str = "microsoft/deberta-v3-base",
    device: str = "cuda",
    dtype: str = "fp32",
    seq_lens: Optional[List[int]] = None,
    num_warmup: int = 20,
    num_timed: int = 50,
) -> Tuple[List[TimingDistribution], PowerLawFit]:
    """Benchmark model forward pass across sequence lengths n <= 512 and fit power law."""
    if seq_lens is None:
        seq_lens = [16, 32, 64, 128, 256, 384, 512]

    logger.info(f"Loading {model_name} ({dtype}) for token scaling benchmark...")
    try:
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
    except Exception:
        cfg = AutoConfig.from_pretrained(model_name)
    cfg.num_labels = 2
    model = AutoModelForSequenceClassification.from_config(cfg).to(device)
    if dtype == "fp16":
        model = model.half()
    model.eval()

    distributions: List[TimingDistribution] = []
    measured_p50s: List[float] = []

    for n in seq_lens:
        dummy_input = torch.randint(100, 1000, (1, n), dtype=torch.long, device=device)
        dummy_mask = torch.ones((1, n), dtype=torch.long, device=device)

        def forward_fn() -> None:
            with torch.no_grad():
                _ = model(input_ids=dummy_input, attention_mask=dummy_mask)

        dist = harness.run_benchmark(
            name=f"scale_{model_name.split('/')[-1]}_{dtype}_n{n}",
            fn=forward_fn,
            num_warmup=num_warmup,
            num_timed=num_timed,
            device=device,
            dtype=dtype,
        )
        distributions.append(dist)
        measured_p50s.append(dist.p50_ms)

    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    fit = fit_power_law(seq_lens, measured_p50s, seed=harness.seed)
    return distributions, fit


def verify_roberta_architectural_limit(
    device: str = "cpu",
    test_lengths: Optional[List[int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Empirically test RoBERTa-base positional embedding limit (Claim C042, Audit Issue M-3).
    
    Standard RoBERTa-base has max_position_embeddings = 514 (512 positions + 2 offset).
    Evaluating at n in {768, 1024} throws an unrecoverable index or size error.
    """
    if test_lengths is None:
        test_lengths = [256, 512, 768, 1024]

    results: Dict[int, Dict[str, Any]] = {}
    model_name = "roberta-base"

    try:
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    except Exception:
        config = AutoConfig.from_pretrained(model_name)
    max_pos = config.max_position_embeddings

    try:
        model = AutoModelForSequenceClassification.from_config(config).to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return {n: {"status": "load_failed", "error": str(e)} for n in test_lengths}

    for n in test_lengths:
        dummy_input = torch.randint(100, 1000, (1, n), dtype=torch.long, device=device)
        dummy_mask = torch.ones((1, n), dtype=torch.long, device=device)

        status = "UNKNOWN"
        error_type = ""
        error_msg = ""
        exception_trace = ""

        try:
            with torch.no_grad():
                _ = model(input_ids=dummy_input, attention_mask=dummy_mask)
            status = "SUCCESS"
        except Exception as ex:
            status = "CRASHED_EXPECTED"
            error_type = type(ex).__name__
            error_msg = str(ex)
            exception_trace = traceback.format_exc()

        results[n] = {
            "seq_len": n,
            "max_position_embeddings": max_pos,
            "exceeds_limit": n > (max_pos - 2),
            "status": status,
            "error_type": error_type,
            "error_msg": error_msg,
            "traceback_snippet": exception_trace.splitlines()[-2:] if exception_trace else [],
        }

    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


class SyntheticSelfAttention(nn.Module):
    """Standard Multi-Head Attention layer to benchmark long-context scaling (n up to 4096)."""

    def __init__(self, d_model: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard quadratic attention: Q K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.out_proj(out)


def sweep_long_context_scaling(
    harness: BenchmarkHarness,
    device: str = "cuda",
    seq_lens: Optional[List[int]] = None,
    d_model: int = 768,
    num_warmup: int = 10,
    num_timed: int = 30,
) -> Tuple[List[TimingDistribution], PowerLawFit]:
    """Benchmark self-attention up to n=4096 to observe crossover toward quadratic scaling."""
    if seq_lens is None:
        seq_lens = [512, 1024, 2048, 4096]

    layer = SyntheticSelfAttention(d_model=d_model).to(device)
    layer.eval()

    distributions: List[TimingDistribution] = []
    measured_p50s: List[float] = []

    for n in seq_lens:
        dummy_x = torch.randn(1, n, d_model, device=device)

        def forward_fn() -> None:
            with torch.no_grad():
                _ = layer(dummy_x)

        dist = harness.run_benchmark(
            name=f"long_ctx_attn_d{d_model}_n{n}",
            fn=forward_fn,
            num_warmup=num_warmup,
            num_timed=num_timed,
            device=device,
            dtype="fp32",
        )
        distributions.append(dist)
        measured_p50s.append(dist.p50_ms)

    del layer
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    fit = fit_power_law(seq_lens, measured_p50s, seed=harness.seed)
    return distributions, fit
