"""Silence Shortcut Audit and Control Engine (Red-Flag RF1).

Implements:
1. Extraction of silence duration features (leading silence, trailing silence,
   total silence, duration) across audio utterances.
2. Silence-only classifier to establish the shortcut baseline EER (Müller et al., 2021).
3. Audio silence-trimming transformation pipeline.
4. Comparative reporting (raw vs trimmed audio performance).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from ares.audio.features import measure_silence_durations, trim_silence
from ares.audio.speaker import compute_eer


class SilenceShortcutAuditor:
    """Audits whether audio deepfake detectors exploit silence artifacts (RF1)."""
    def __init__(self):
        self.classifier = LogisticRegression(random_state=42, max_iter=500)
        self.is_fitted = False

    @staticmethod
    def extract_silence_feature_vector(waveform: torch.Tensor, sample_rate: int = 16000) -> np.ndarray:
        """Extract a 5-dimensional silence feature vector from waveform."""
        stats = measure_silence_durations(waveform, sample_rate=sample_rate)
        return np.array([
            stats["leading_silence_sec"],
            stats["trailing_silence_sec"],
            stats["total_silence_sec"],
            stats["total_duration_sec"],
            stats["speech_duration_sec"],
        ], dtype=np.float32)

    def fit_silence_baseline(self, X_silence: np.ndarray, y_labels: np.ndarray) -> "SilenceShortcutAuditor":
        """Fit a linear classifier exclusively on silence duration features."""
        self.classifier.fit(X_silence, y_labels)
        self.is_fitted = True
        return self

    def evaluate_silence_baseline(
        self,
        X_silence: np.ndarray,
        y_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate how well silence duration alone predicts the spoofing label."""
        if not self.is_fitted:
            raise RuntimeError("Auditor must be fitted first")
            
        prob_spoof = self.classifier.predict_proba(X_silence)[:, 1]
        auc = float(roc_auc_score(y_labels, prob_spoof))
        
        # Split bonafide (0) and spoof (1) scores
        # Note: in ASVspoof scoring, higher score = more likely bonafide
        # So we use 1 - prob_spoof as bonafide score
        bonafide_scores = 1.0 - prob_spoof[y_labels == 0]
        spoof_scores = 1.0 - prob_spoof[y_labels == 1]
        eer, thresh = compute_eer(bonafide_scores, spoof_scores)
        
        return {
            "silence_only_auc": auc,
            "silence_only_eer": eer,
            "silence_threshold": thresh,
        }
