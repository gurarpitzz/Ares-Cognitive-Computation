"""Topic & Domain Classification module for ARES Geopolitical & Contextual Intelligence.

Implements Eq. (4) and Section V-B:
    P(c|x) = exp(z_c) / sum_j exp(z_j)
Supports zero-shot NLI classification, dataset-source domain mapping,
and Table V regeneration (Claim C008).
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STANDARD_DOMAINS = [
    "Politics",
    "Health",
    "Conflict",
    "Disaster",
    "Economy",
    "Technology",
    "Science",
    "Entertainment",
    "Other",
]

# Source-dataset prior domain rules
DATASET_DOMAIN_RULES = {
    "coaid": "Health",
    "constraint": "Health",
    "covid": "Health",
    "liar": "Politics",
    "politifact": "Politics",
    "fever": "Other",
    "gossipcop": "Entertainment",
    "celebdf": "Entertainment",
}


@dataclass
class TopicPrediction:
    """Represents a topic classification output."""
    predicted_topic: str
    confidence: float
    all_scores: Dict[str, float]
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_topic": self.predicted_topic,
            "confidence": round(self.confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
            "method": self.method,
        }


class TopicClassifier:
    """Multi-domain classifier combining source metadata rules and zero-shot NLI."""

    def __init__(
        self,
        candidate_labels: Optional[List[str]] = None,
        model_name: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
    ):
        self.candidate_labels = candidate_labels or STANDARD_DOMAINS
        self.model_name = model_name
        self._device = device
        self._pipeline = None

    def _get_device(self) -> int:
        if self._device is not None:
            return 0 if "cuda" in self._device else -1
        import torch
        return 0 if torch.cuda.is_available() else -1

    def _load_pipeline(self):
        if self._pipeline is None:
            import glob, os, torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            # Search in local HF hub cache first for full offline replay
            cache_pattern = os.path.expanduser(f"~/.cache/huggingface/hub/models--{self.model_name.replace('/', '--')}/snapshots/*")
            snapshots = glob.glob(cache_pattern)

            dev = self._get_device()
            if snapshots:
                local_dir = snapshots[0]
                logger.info("Loading TopicClassifier from local snapshot: %s", local_dir)
                tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(local_dir, local_files_only=True)
                if dev >= 0:
                    model = model.to("cuda")
                self._pipeline = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=dev)
            else:
                logger.info("Local snapshot not found for %s, falling back to pipeline download/check", self.model_name)
                self._pipeline = pipeline("zero-shot-classification", model=self.model_name, device=dev)

        return self._pipeline

    def classify_with_rules(self, text: str, dataset_source: Optional[str] = None) -> Optional[TopicPrediction]:
        """Apply deterministic metadata rules if available."""
        if dataset_source:
            norm_source = dataset_source.lower().strip()
            for key, domain in DATASET_DOMAIN_RULES.items():
                if key in norm_source:
                    scores = {d: (1.0 if d == domain else 0.0) for d in self.candidate_labels}
                    return TopicPrediction(
                        predicted_topic=domain,
                        confidence=1.0,
                        all_scores=scores,
                        method="dataset_source_rule",
                    )
        return None

    def classify(
        self,
        text: str,
        dataset_source: Optional[str] = None,
        candidate_labels: Optional[List[str]] = None,
        threshold: float = 0.40,
    ) -> TopicPrediction:
        """Classify text into a topic domain using rules first, then zero-shot NLI."""
        # 1. Check metadata rule first
        rule_pred = self.classify_with_rules(text, dataset_source)
        if rule_pred is not None:
            return rule_pred

        # 2. Zero-shot NLI classification
        labels = candidate_labels or self.candidate_labels
        pipe = self._load_pipeline()
        result = pipe(text, candidate_labels=labels)

        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        score_dict = {lbl: float(sc) for lbl, sc in zip(result["labels"], result["scores"])}

        # If below pre-registered confidence threshold, assign "Other"
        if top_score < threshold and "Other" in labels:
            final_label = "Other"
        else:
            final_label = top_label

        return TopicPrediction(
            predicted_topic=final_label,
            confidence=top_score,
            all_scores=score_dict,
            method="zero_shot_nli",
        )


def regenerate_table_v(classifier: Optional[TopicClassifier] = None) -> List[Dict[str, Any]]:
    """Regenerate Table V (Examples of Topic Classification) from actual model outputs.

    Corresponds to Claim C008 in the claims ledger.
    """
    if classifier is None:
        classifier = TopicClassifier(candidate_labels=["Economy", "Technology", "Science", "Health", "Politics", "Entertainment"])

    monograph_samples = [
        "The economy is facing inflation due to the pandemic.",
        "Tesla's new car model will be released next month.",
        "New breakthroughs in quantum computing are promising.",
    ]

    rows = []
    for sample in monograph_samples:
        pred = classifier.classify(sample, candidate_labels=["Economy", "Technology", "Science", "Health", "Politics", "Entertainment"])
        rows.append({
            "Text Sample": f'"{sample}"',
            "Predicted Topic": pred.predicted_topic,
            "Confidence": round(pred.confidence, 4),
            "Method": pred.method,
        })

    return rows
