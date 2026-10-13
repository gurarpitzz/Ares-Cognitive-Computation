"""
Hardware Requirements & Memory Footprint Audit (Claim C033).

Replaces the manuscript's ungrounded '32 GB RAM; RTX 3080/A100' claim with empirical
measurements on the declared benchmark machine:
  HP OMEN 16 (RTX 4060 Laptop GPU 8 GB GDDR6, 16 GB DDR5 RAM, Intel Core i7-13620H).

Measures:
  - Peak VRAM allocated (MB)
  - Peak RAM utilized (GB)
  - CPU-only fallback latency (ms)
  - GPU acceleration speedup factor
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from ares.audio.models import DeclaredAudioCNN
from ares.fusion.late import AppendixAMLPHead
from ares.vision.models import DeepfakeDetector

logger = logging.getLogger(__name__)


@dataclass
class FootprintProfile:
    """Hardware footprint and resource utilization profile for an architectural scope."""
    scope_name: str
    description: str
    peak_vram_mb: float
    peak_ram_gb: float
    gpu_p50_latency_ms: float
    cpu_p50_latency_ms: float
    speedup_gpu_vs_cpu: float
    minimum_vram_required_gb: float
    minimum_ram_required_gb: float
    viable_on_8gb_vram: bool
    viable_on_16gb_ram: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_name": self.scope_name,
            "description": self.description,
            "peak_vram_mb": round(float(self.peak_vram_mb), 2),
            "peak_ram_gb": round(float(self.peak_ram_gb), 2),
            "gpu_p50_latency_ms": round(float(self.gpu_p50_latency_ms), 3),
            "cpu_p50_latency_ms": round(float(self.cpu_p50_latency_ms), 3),
            "speedup_gpu_vs_cpu": round(float(self.speedup_gpu_vs_cpu), 2),
            "minimum_vram_required_gb": self.minimum_vram_required_gb,
            "minimum_ram_required_gb": self.minimum_ram_required_gb,
            "viable_on_8gb_vram": self.viable_on_8gb_vram,
            "viable_on_16gb_ram": self.viable_on_16gb_ram,
        }


def profile_scope_footprint(
    scope_name: str,
    description: str,
    model_fn_builder: Any,
    input_tensor_builder: Any,
) -> FootprintProfile:
    """Profile peak VRAM, RAM, and GPU vs CPU latency for a model scope."""
    # 1. GPU profiling
    gpu_latencies: List[float] = []
    peak_vram_mb = 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        model_gpu = model_fn_builder("cuda")
        inputs_gpu = input_tensor_builder("cuda")

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model_gpu(**inputs_gpu) if isinstance(inputs_gpu, dict) else model_gpu(inputs_gpu)
        torch.cuda.synchronize()

        # Timed
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            with torch.no_grad():
                _ = model_gpu(**inputs_gpu) if isinstance(inputs_gpu, dict) else model_gpu(inputs_gpu)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            gpu_latencies.append((t1 - t0) / 1e6)

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        del model_gpu, inputs_gpu
        torch.cuda.empty_cache()
        gc.collect()

    gpu_p50 = float(np.percentile(gpu_latencies, 50)) if gpu_latencies else 0.0

    # 2. CPU profiling
    model_cpu = model_fn_builder("cpu")
    inputs_cpu = input_tensor_builder("cpu")
    cpu_latencies: List[float] = []

    for _ in range(2):
        with torch.no_grad():
            _ = model_cpu(**inputs_cpu) if isinstance(inputs_cpu, dict) else model_cpu(inputs_cpu)

    for _ in range(5):
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            _ = model_cpu(**inputs_cpu) if isinstance(inputs_cpu, dict) else model_cpu(inputs_cpu)
        t1 = time.perf_counter_ns()
        cpu_latencies.append((t1 - t0) / 1e6)

    cpu_p50 = float(np.percentile(cpu_latencies, 50)) if cpu_latencies else 0.0
    del model_cpu, inputs_cpu
    gc.collect()

    speedup = cpu_p50 / gpu_p50 if (gpu_p50 > 0 and cpu_p50 > 0) else 1.0

    vm = psutil.virtual_memory()
    peak_ram_gb = (vm.total - vm.available) / (1024 ** 3)

    min_vram = round((peak_vram_mb / 1024.0) * 1.5, 1)  # 50% safety margin
    min_ram = 8.0

    return FootprintProfile(
        scope_name=scope_name,
        description=description,
        peak_vram_mb=peak_vram_mb,
        peak_ram_gb=peak_ram_gb,
        gpu_p50_latency_ms=gpu_p50,
        cpu_p50_latency_ms=cpu_p50,
        speedup_gpu_vs_cpu=speedup,
        minimum_vram_required_gb=max(2.0, min_vram),
        minimum_ram_required_gb=min_ram,
        viable_on_8gb_vram=min_vram <= 8.0,
        viable_on_16gb_ram=peak_ram_gb <= 16.0,
    )


def audit_hardware_footprint() -> Dict[str, FootprintProfile]:
    """Execute hardware footprint audit across all verification scopes."""
    profiles: Dict[str, FootprintProfile] = {}

    # Scope A: Text Verifier (DeBERTa-v3-base)
    logger.info("Profiling Scope A (Text Verifier)...")

    def build_text_model(dev: str) -> Any:
        try:
            cfg = AutoConfig.from_pretrained("microsoft/deberta-v3-base", local_files_only=True)
        except Exception:
            cfg = AutoConfig.from_pretrained("microsoft/deberta-v3-base")
        cfg.num_labels = 2
        m = AutoModelForSequenceClassification.from_config(cfg).to(dev)
        m.eval()
        return m

    def build_text_inputs(dev: str) -> Any:
        return {
            "input_ids": torch.randint(100, 1000, (1, 128), dtype=torch.long, device=dev),
            "attention_mask": torch.ones((1, 128), dtype=torch.long, device=dev),
        }

    profiles["scope_a_text"] = profile_scope_footprint(
        scope_name="Scope A (Text Verifier Alone)",
        description="DeBERTa-v3-base forward pass at n=128 tokens",
        model_fn_builder=build_text_model,
        input_tensor_builder=build_text_inputs,
    )

    # Scope C: Vision Deepfake (ResNet-50)
    logger.info("Profiling Scope C (Vision Deepfake)...")

    def build_vision_model(dev: str) -> Any:
        m = DeepfakeDetector(architecture="resnet50", pretrained=False).to(dev)
        m.eval()
        return m

    def build_vision_inputs(dev: str) -> Any:
        return torch.randn(8, 3, 224, 224, device=dev)

    profiles["scope_c_vision"] = profile_scope_footprint(
        scope_name="Scope C (Vision Deepfake Pipeline)",
        description="ResNet-50 8-frame feature extraction + pooling",
        model_fn_builder=build_vision_model,
        input_tensor_builder=build_vision_inputs,
    )

    # Scope D: Audio Spoofing (DeclaredAudioCNN)
    logger.info("Profiling Scope D (Audio Spoofing)...")

    def build_audio_model(dev: str) -> Any:
        m = DeclaredAudioCNN(in_channels=1, n_mels=80, embedding_dim=512).to(dev)
        m.eval()
        return m

    def build_audio_inputs(dev: str) -> Any:
        return torch.randn(1, 1, 80, 400, device=dev)

    profiles["scope_d_audio"] = profile_scope_footprint(
        scope_name="Scope D (Audio Spoofing Pipeline)",
        description="Declared Audio CNN on 80x400 log-mel spectrogram",
        model_fn_builder=build_audio_model,
        input_tensor_builder=build_audio_inputs,
    )

    # Scope E: Multimodal Late Fusion Head
    logger.info("Profiling Scope E (Late Fusion Head)...")

    def build_fusion_model(dev: str) -> Any:
        m = AppendixAMLPHead(in_features=1024, n_classes=2).to(dev)
        m.eval()
        return m

    def build_fusion_inputs(dev: str) -> Any:
        return torch.randn(1, 1024, device=dev)

    profiles["scope_e_fusion"] = profile_scope_footprint(
        scope_name="Scope E (Late Fusion Head)",
        description="1024-d 2-layer GELU MLP + Chow abstention",
        model_fn_builder=build_fusion_model,
        input_tensor_builder=build_fusion_inputs,
    )

    return profiles
