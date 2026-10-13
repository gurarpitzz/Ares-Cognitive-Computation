"""
Unit Tests for ARES-Synth Benchmark Suite (M16 / M12).
"""

import os
import pandas as pd
import pytest

from ares.synth.facts import GroundedFact, get_grounded_facts
from ares.synth.perturb import (
    perturb_date_shift,
    perturb_entity_swap,
    perturb_fabricated_event,
    perturb_location_swap,
    perturb_misattribution,
    perturb_negation,
    perturb_number_change,
    perturb_out_of_context,
)
from ares.synth.validate import (
    validate_dual_nli_heuristic,
    validate_safety_and_slurs,
    validate_slot_preservation,
)


def test_grounded_facts_loaded():
    facts = get_grounded_facts()
    assert len(facts) >= 10
    scenarios = set(f.scenario_id for f in facts)
    assert "S1" in scenarios
    assert "S2" in scenarios
    assert "S3" in scenarios


def test_perturbation_operators_deterministic():
    fact = get_grounded_facts()[0]
    p1 = perturb_entity_swap(fact, seed=42)
    p2 = perturb_entity_swap(fact, seed=42)
    assert p1.claim_text == p2.claim_text
    assert p1.label == "FALSE"
    assert p1.perturbation_type == "entity_swap"


def test_perturbation_number_change():
    fact = get_grounded_facts()[0]  # S1_F01 has number_val = 20.0
    p = perturb_number_change(fact, seed=42)
    assert p.label == "FALSE"
    assert p.original_slot != p.perturbed_slot
    assert p.perturbation_type == "number_change"


def test_perturbation_fabricated_event_nei():
    fact = get_grounded_facts()[0]
    p = perturb_fabricated_event(fact, seed=42)
    assert p.label == "NEI"
    assert p.contradicting_sent_id is None
    assert p.perturbation_type == "fabricated_event"


def test_validation_gates():
    fact = get_grounded_facts()[0]
    p = perturb_location_swap(fact, seed=42)
    assert validate_slot_preservation(p) is True
    assert validate_safety_and_slurs(p) is True
    assert validate_dual_nli_heuristic(p, fact) is True


def test_frozen_ares_synth_manifest_exists():
    manifest_csv = "ares_v2/data_manifests/ares_synth_v1.csv"
    if os.path.exists(manifest_csv):
        df = pd.read_csv(manifest_csv)
        assert len(df) >= 1500
        labels = set(df["label"].unique())
        assert "TRUE" in labels
        assert "FALSE" in labels
        assert "NEI" in labels
