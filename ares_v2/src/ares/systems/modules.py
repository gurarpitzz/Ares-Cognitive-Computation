"""
Per-Module Latency Benchmarks for ARES Systems (Claims C016, C017).

Standardized measurement across all multimodal and cognitive verification layers:
1. Text Verifier: DeBERTa-v3-base forward pass at n=128 (FP32 and FP16).
2. NLI Passage Scorer: Cross-encoder scoring per-passage and per-claim (M=5 passages).
3. Evidence Retrieval: BM25 index query and top-k ranking over local corpus.
4. Symbolic Logic Checker: Numerical, temporal, and consistency rule evaluation.
5. Visual Deepfake Detector: ResNet-50 feature extraction on video frames (16 frames at 1-2 FPS).
6. Audio Spoofing Detector: 4-second waveform -> 80-channel log-mel -> DeclaredAudioCNN.
7. Geopolitical Toponym & NER: Entity extraction and coordinate resolution.
8. Late Fusion Head: 1024-d 2-layer GELU MLP + Chow abstention calculation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from ares.audio.features import LogMelSpectrogramExtractor
from ares.audio.models import DeclaredAudioCNN
from ares.decision.abstain import ChowAbstention
from ares.fusion.late import AppendixAMLPHead
from ares.systems.bench import BenchmarkHarness, TimingDistribution
from ares.vision.models import DeepfakeDetector

logger = logging.getLogger(__name__)


class SyntheticBM25Index:
    """Fast, deterministic BM25 retriever for local index benchmarking."""

    def __init__(self, num_docs: int = 1000, avg_doc_len: int = 150, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.num_docs = num_docs
        vocab = [f"token_{i}" for i in range(5000)]
        self.docs = [
            " ".join(self.rng.choice(vocab, size=self.rng.integers(50, avg_doc_len * 2)))
            for _ in range(num_docs)
        ]
        # Inverted index
        self.index: Dict[str, List[int]] = {}
        for doc_id, doc in enumerate(self.docs):
            for word in set(doc.split()):
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append(doc_id)

    def query(self, query_str: str, top_k: int = 5) -> List[int]:
        scores: Dict[int, float] = {}
        for token in query_str.split():
            doc_ids = self.index.get(token, [])
            for doc_id in doc_ids:
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [doc_id for doc_id, _ in ranked]


class SyntheticLogicChecker:
    """Symbolic rule evaluator for quantities, temporal constraints, and entities."""

    def __init__(self) -> None:
        self.rules = [
            lambda claim: "died" in claim and "survived" in claim,
            lambda claim: any(c.isdigit() for c in claim),
            lambda claim: len(claim) > 10,
        ]

    def verify(self, claim_text: str) -> Dict[str, Any]:
        results = [r(claim_text) for r in self.rules]
        contradiction = results[0]
        has_quant = results[1]
        return {
            "has_contradiction": contradiction,
            "has_quantities": has_quant,
            "valid": not contradiction,
            "logic_score": 0.0 if contradiction else (0.8 if has_quant else 0.5),
        }


def benchmark_text_verifier(
    harness: BenchmarkHarness,
    device: str = "cuda",
    dtype: str = "fp32",
    seq_len: int = 128,
    num_warmup: int = 30,
    num_timed: int = 100,
    model_name: str = "microsoft/deberta-v3-base",
) -> TimingDistribution:
    """Benchmark text verifier forward pass (DeBERTa-v3-base, n=128)."""
    logger.info(f"Loading {model_name} for text verifier benchmark...")
    try:
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
    except Exception:
        cfg = AutoConfig.from_pretrained(model_name)
    cfg.num_labels = 2
    model = AutoModelForSequenceClassification.from_config(cfg).to(device)
    if dtype == "fp16":
        model = model.half()
    model.eval()

    inputs = {
        "input_ids": torch.randint(100, 1000, (1, seq_len), dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, seq_len), dtype=torch.long, device=device),
    }

    def forward_fn() -> None:
        with torch.no_grad():
            _ = model(**inputs)

    result = harness.run_benchmark(
        name=f"text_verifier_{model_name.split('/')[-1]}_{dtype}_n{seq_len}",
        fn=forward_fn,
        num_warmup=num_warmup,
        num_timed=num_timed,
        device=device,
        dtype=dtype,
    )

    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def benchmark_nli_passage_scoring(
    harness: BenchmarkHarness,
    device: str = "cuda",
    dtype: str = "fp32",
    num_passages: int = 5,
    seq_len: int = 128,
    num_warmup: int = 20,
    num_timed: int = 80,
    model_name: str = "typeform/distilbert-base-uncased-mnli",
) -> TimingDistribution:
    """Benchmark NLI cross-encoder scoring across M=5 retrieved passages."""
    logger.info(f"Loading {model_name} for NLI benchmark...")
    try:
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
    except Exception:
        cfg = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_config(cfg).to(device)
    if dtype == "fp16":
        model = model.half()
    model.eval()

    inputs = {
        "input_ids": torch.randint(100, 1000, (num_passages, seq_len), dtype=torch.long, device=device),
        "attention_mask": torch.ones((num_passages, seq_len), dtype=torch.long, device=device),
    }

    def forward_fn() -> None:
        with torch.no_grad():
            _ = model(**inputs)

    result = harness.run_benchmark(
        name=f"nli_cross_encoder_{dtype}_M{num_passages}",
        fn=forward_fn,
        num_warmup=num_warmup,
        num_timed=num_timed,
        device=device,
        dtype=dtype,
    )

    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def benchmark_bm25_retrieval(
    harness: BenchmarkHarness,
    corpus_size: int = 1000,
    top_k: int = 5,
    num_warmup: int = 20,
    num_timed: int = 100,
) -> TimingDistribution:
    """Benchmark BM25 passage retrieval over local index."""
    index = SyntheticBM25Index(num_docs=corpus_size)
    query_str = "token_42 token_100 token_305 token_999"

    def query_fn() -> None:
        _ = index.query(query_str, top_k=top_k)

    return harness.run_benchmark(
        name=f"bm25_retrieval_N{corpus_size}_k{top_k}",
        fn=query_fn,
        num_warmup=num_warmup,
        num_timed=num_timed,
        device="cpu",
        dtype="fp32",
    )


def benchmark_logic_checker(
    harness: BenchmarkHarness,
    num_warmup: int = 50,
    num_timed: int = 200,
) -> TimingDistribution:
    """Benchmark symbolic logic constraint verification."""
    checker = SyntheticLogicChecker()
    claim = "Casualties reported at 42 people on 2026-10-12 while 15 survived."

    def logic_fn() -> None:
        _ = checker.verify(claim)

    return harness.run_benchmark(
        name="symbolic_logic_checker",
        fn=logic_fn,
        num_warmup=num_warmup,
        num_timed=num_timed,
        device="cpu",
        dtype="fp32",
    )


def benchmark_video_deepfake(
    harness: BenchmarkHarness,
    device: str = "cuda",
    num_frames: int = 16,
    num_warmup: int = 10,
    num_timed: int = 50,
) -> TimingDistribution:
    """Benchmark video deepfake feature extraction (ResNet-50, 16 frames at 1-2 FPS)."""
    detector = DeepfakeDetector(architecture="resnet50", pretrained=False).to(device)
    detector.eval()

    dummy_frames = torch.randn(num_frames, 3, 224, 224, device=device)

    def forward_fn() -> None:
        with torch.no_grad():
            features = detector.extract_features(dummy_frames)
            _ = torch.mean(features, dim=0, keepdim=True)

    result = harness.run_benchmark(
        name=f"video_deepfake_resnet50_frames{num_frames}",
        fn=forward_fn,
        num_warmup=num_warmup,
        num_timed=num_timed,
        device=device,
        dtype="fp32",
    )

    del detector
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def benchmark_audio_spoofing(
    harness: BenchmarkHarness,
    device: str = "cuda",
    sample_rate: int = 16000,
    duration_s: float = 4.0,
    num_warmup: int = 20,
    num_timed: int = 100,
) -> TimingDistribution:
    """Benchmark audio spoofing detector (4-second waveform -> log-mel -> DeclaredAudioCNN)."""
    model = DeclaredAudioCNN(in_channels=1, n_mels=80, embedding_dim=512).to(device)
    model.eval()

    num_samples = int(sample_rate * duration_s)
    raw_waveform = torch.randn(1, num_samples, dtype=torch.float32, device=device)
    extractor = LogMelSpectrogramExtractor(sample_rate=sample_rate, n_mels=80)

    def audio_pipeline_fn() -> None:
        mel = extractor(raw_waveform)
        # Ensure shape [1, 1, n_mels, time]
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        with torch.no_grad():
            _ = model(mel)

    result = harness.run_benchmark(
        name=f"audio_spoof_declared_cnn_{int(duration_s)}s",
        fn=audio_pipeline_fn,
        num_warmup=num_warmup,
        num_timed=num_timed,
        device=device,
        dtype="fp32",
    )

    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def benchmark_toponym_geo(
    harness: BenchmarkHarness,
    num_warmup: int = 20,
    num_timed: int = 100,
) -> TimingDistribution:
    """Benchmark toponym entity extraction and gazetteer lookup."""
    gazetteer = {
        "new york": (40.7128, -74.0060),
        "assam": (26.2006, 92.9376),
        "tokyo": (35.6762, 139.6503),
        "beijing": (39.9042, 116.4074),
        "london": (51.5074, -0.1278),
    }
    claim = "Severe flooding continues across Assam and neighboring regions."

    def geo_fn() -> None:
        words = claim.lower().split()
        resolved = []
        for word in words:
            if word in gazetteer:
                resolved.append(gazetteer[word])
        _ = resolved

    return harness.run_benchmark(
        name="toponym_geotag_lookup",
        fn=geo_fn,
        num_warmup=num_warmup,
        num_timed=num_timed,
        device="cpu",
        dtype="fp32",
    )


def benchmark_late_fusion_head(
    harness: BenchmarkHarness,
    device: str = "cuda",
    num_warmup: int = 50,
    num_timed: int = 500,
) -> TimingDistribution:
    """Benchmark 1024-d 2-layer GELU MLP + Chow abstention calculation."""
    mlp = AppendixAMLPHead(in_features=1024, n_classes=2).to(device)
    mlp.eval()
    abstention = ChowAbstention(c_fp=1.0, c_fn=1.0, c_a=0.2)

    features = torch.randn(1, 1024, device=device)

    def fusion_fn() -> None:
        with torch.no_grad():
            logits = mlp(features)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            log_odds = float(np.log(np.clip(probs[1] / max(probs[0], 1e-7), 1e-7, 1e7)))
            _ = abstention.decide(log_odds)

    result = harness.run_benchmark(
        name="late_fusion_1024d_gelu_chow",
        fn=fusion_fn,
        num_warmup=num_warmup,
        num_timed=num_timed,
        device=device,
        dtype="fp32",
    )

    del mlp
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def benchmark_all_modules(
    harness: BenchmarkHarness,
    device: str = "cuda",
) -> Dict[str, TimingDistribution]:
    """Execute complete suite of per-module benchmarks."""
    results: Dict[str, TimingDistribution] = {}

    logger.info("Running M13.2 Per-Module Benchmarks...")
    results["text_deberta_fp32"] = benchmark_text_verifier(harness, device=device, dtype="fp32")
    results["text_deberta_fp16"] = benchmark_text_verifier(harness, device=device, dtype="fp16")
    results["nli_scoring"] = benchmark_nli_passage_scoring(harness, device=device)
    results["bm25_retrieval"] = benchmark_bm25_retrieval(harness)
    results["logic_checker"] = benchmark_logic_checker(harness)
    results["video_deepfake"] = benchmark_video_deepfake(harness, device=device)
    results["audio_spoofing"] = benchmark_audio_spoofing(harness, device=device)
    results["toponym_geo"] = benchmark_toponym_geo(harness)
    results["late_fusion"] = benchmark_late_fusion_head(harness, device=device)

    return results
