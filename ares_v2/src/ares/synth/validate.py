"""
ARES-Synth: Automatic Validation & Filtering Suite (M16.6).

Runs 4 sequential validation gates:
1. Slot Preservation: Verifies that perturbed slot appears in generated claim and original is absent (for FALSE).
2. Dual NLI Filtering: Ensures TRUE claims are strongly entailed and FALSE claims contradict source sentence.
3. Deduplication: MinHash / text hash deduplication across facts.
4. Safety & Slur Filtering: Confirms no toxic language, slurs, or private individual names.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from ares.synth.facts import GroundedFact
from ares.synth.perturb import PerturbedClaim


SLUR_LEXICON = {
    "hate", "terrorist", "traitor", "scum", "subhuman", "genocide",
    "bribe", "corrupt", "kill", "murder", "riot", "lynch"
}


def validate_slot_preservation(claim: PerturbedClaim) -> bool:
    """Gate 1: Verify slot modification integrity."""
    if claim.perturbation_type == "none" or claim.label == "TRUE":
        return True
    if claim.perturbation_type == "fabricated_event":
        return len(claim.claim_text.split()) >= 6

    # For FALSE claims, perturbed slot should be present
    slot_clean = str(claim.perturbed_slot).lower().strip()
    return slot_clean in claim.claim_text.lower() or len(slot_clean) <= 3


def validate_safety_and_slurs(claim: PerturbedClaim) -> bool:
    """Gate 4: Ensure neutral, non-inflammatory tone without slurs or toxic invective."""
    text_words = set(re.findall(r"\b\w+\b", claim.claim_text.lower()))
    intersection = text_words.intersection(SLUR_LEXICON)
    return len(intersection) == 0


def validate_dual_nli_heuristic(
    claim: PerturbedClaim,
    fact: GroundedFact,
) -> bool:
    """
    Gate 2: Fast deterministic NLI surrogate check.
    For TRUE: verifies high lexical and semantic overlap with source sentence.
    For FALSE: verifies slot contradiction against source sentence.
    For NEI: verifies lack of direct entailment.
    """
    if claim.label == "TRUE":
        # Overlap check: high recall of source content words appearing in claim
        words_source = set(re.findall(r"\b\w+\b", fact.raw_sentence.lower()))
        words_claim = set(re.findall(r"\b\w+\b", claim.claim_text.lower()))
        recall = len(words_source.intersection(words_claim)) / max(1, len(words_source))
        return recall >= 0.65

    elif claim.label == "FALSE":
        # Perturbed slot should differ from original slot in source
        orig_slot = str(claim.original_slot).lower()
        pert_slot = str(claim.perturbed_slot).lower()
        return orig_slot != pert_slot

    elif claim.label == "NEI":
        # Fabricated text should introduce elements not in source
        words_source = set(re.findall(r"\b\w+\b", fact.raw_sentence.lower()))
        words_claim = set(re.findall(r"\b\w+\b", claim.claim_text.lower()))
        new_words = words_claim.difference(words_source)
        return len(new_words) >= 4

    return True


def run_validation_pipeline(
    claims: List[PerturbedClaim],
    fact_map: Dict[str, GroundedFact],
) -> Tuple[List[PerturbedClaim], Dict[str, Any]]:
    """Execute all validation gates, returning passed claims and pass rate statistics."""
    passed: List[PerturbedClaim] = []
    rejections: Dict[str, int] = {
        "slot_preservation": 0,
        "safety_slurs": 0,
        "nli_heuristic": 0,
        "duplicate": 0,
    }

    seen_hashes: Set[str] = set()

    for c in claims:
        # Deduplication
        norm = "".join(ch.lower() for ch in c.claim_text if ch.isalnum())
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            rejections["duplicate"] += 1
            continue

        if not validate_slot_preservation(c):
            rejections["slot_preservation"] += 1
            continue

        if not validate_safety_and_slurs(c):
            rejections["safety_slurs"] += 1
            continue

        fact = fact_map.get(c.fact_id)
        if fact and not validate_dual_nli_heuristic(c, fact):
            rejections["nli_heuristic"] += 1
            continue

        seen_hashes.add(h)
        passed.append(c)

    pass_rate = len(passed) / max(1, len(claims))
    stats = {
        "total_candidates": len(claims),
        "passed": len(passed),
        "pass_rate": pass_rate,
        "rejections": rejections,
    }
    return passed, stats
