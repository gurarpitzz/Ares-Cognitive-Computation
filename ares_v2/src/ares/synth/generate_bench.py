"""
ARES-Synth: Benchmark Generator & Manifest Freezing Script (M16 / M12).

Constructs ARES-Synth v1:
- Scenarios: S1 (Border Conflict), S2 (Assam Floods), S3 (Indian Election)
- Balanced: >= 600 TRUE, >= 600 FALSE (across 7 perturbation types), >= 300 NEI
- 4 Surface Realization Registers (news, social, forwarded, official)
- Validated via slot preservation, safety, and NLI gates
- Freezes dataset to data/synthetic/ares_synth_v1/ and data_manifests/ares_synth_v1.csv
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ares.synth.facts import GroundedFact, get_grounded_facts
from ares.synth.perturb import (
    PerturbedClaim,
    perturb_date_shift,
    perturb_entity_swap,
    perturb_fabricated_event,
    perturb_location_swap,
    perturb_misattribution,
    perturb_negation,
    perturb_number_change,
    perturb_out_of_context,
)
from ares.synth.validate import run_validation_pipeline


REGISTERS = ["news", "social", "forwarded", "official"]

# Surface realization style templates
STYLE_TEMPLATES = {
    "news": lambda t: f"Breaking News: {t}",
    "social": lambda t: f"Report: {t} #Alert",
    "forwarded": lambda t: f"Urgent update received: {t} Please verify before sharing.",
    "official": lambda t: f"Official bulletin release: {t}",
}


TRUE_TEMPLATES = [
    lambda f: f.raw_sentence,
    lambda f: f"According to official reports, {f.raw_sentence}",
    lambda f: f"In {f.location_name}, {f.raw_sentence}",
    lambda f: f"On {f.date_str}, {f.raw_sentence}",
    lambda f: f"{f.attributed_source} verified that {f.subject} {f.relation} {f.object}.",
    lambda f: f"Official records confirm that {f.raw_sentence}",
    lambda f: f"Disaster and security bulletins confirm: {f.raw_sentence}",
    lambda f: f"Confirmed incident summary: {f.subject} {f.relation} {f.object} ({f.location_name}, {f.date_str}).",
    lambda f: f"Public records from {f.attributed_source} document that {f.raw_sentence}",
    lambda f: f"Verified status update: {f.subject} {f.relation} {f.object} on {f.date_str}.",
    lambda f: f"Field reporting indicates that {f.raw_sentence}",
    lambda f: f"State administration logs indicate: {f.subject} {f.relation} {f.object}.",
    lambda f: f"Formal gazette notice: {f.raw_sentence}",
    lambda f: f"Press release from {f.attributed_source}: {f.raw_sentence}",
    lambda f: f"Documented historical record: {f.subject} {f.relation} {f.object} at {f.location_name}.",
    lambda f: f"Audited institutional communique: {f.raw_sentence}",
    lambda f: f"Comprehensive ground assessment confirms: {f.raw_sentence}",
    lambda f: f"Official inter-agency bulletin: {f.subject} {f.relation} {f.object} as of {f.date_str}.",
    lambda f: f"Archival monitoring logs verify: {f.raw_sentence}",
    lambda f: f"Regional administration briefing reports that {f.raw_sentence}",
    lambda f: f"Validated public interest dispatch: {f.subject} {f.relation} {f.object}.",
    lambda f: f"Government situational briefing: {f.raw_sentence}",
]


def realize_surface_registers(claim_text: str, register: str) -> str:
    """Format structured claim text into target communicative register."""
    formatter = STYLE_TEMPLATES.get(register, lambda t: t)
    return formatter(claim_text)


def build_candidate_pool(facts: List[GroundedFact], target_per_fact: int = 150) -> List[PerturbedClaim]:
    """Generate extensive candidates for TRUE, FALSE (7 perturbation types), and NEI."""
    candidates: List[PerturbedClaim] = []

    perturb_fns = [
        ("entity_swap", perturb_entity_swap),
        ("location_swap", perturb_location_swap),
        ("number_change", perturb_number_change),
        ("date_shift", perturb_date_shift),
        ("negation", perturb_negation),
        ("misattribution", perturb_misattribution),
        ("out_of_context", perturb_out_of_context),
    ]

    for f_idx, fact in enumerate(facts):
        # 1. Generate TRUE claims across diverse templates and registers
        t_count = 0
        for t_idx, template_fn in enumerate(TRUE_TEMPLATES):
            base_text = template_fn(fact)
            for reg in REGISTERS:
                text = realize_surface_registers(base_text, reg)
                c = PerturbedClaim(
                    claim_id=f"{fact.fact_id}_TRUE_{t_idx}_{reg}",
                    fact_id=fact.fact_id,
                    scenario_id=fact.scenario_id,
                    label="TRUE",
                    perturbation_type="none",
                    original_slot="none",
                    perturbed_slot="none",
                    claim_text=text,
                    location_name=fact.location_name,
                    geonames_id=fact.geonames_id,
                    latitude=fact.latitude,
                    longitude=fact.longitude,
                    contradicting_sent_id=None,
                    register=reg,
                )
                candidates.append(c)
                t_count += 1

        # 2. Generate FALSE claims across all 7 perturbation operators
        for op_name, op_fn in perturb_fns:
            for rep in range(12):
                for reg in REGISTERS:
                    seed = 1000 * f_idx + 100 * rep + len(candidates)
                    raw_c = op_fn(fact, seed=seed)
                    text = realize_surface_registers(raw_c.claim_text, reg)
                    raw_c.claim_id = f"{raw_c.claim_id}_{reg}"
                    raw_c.claim_text = text
                    raw_c.register = reg
                    candidates.append(raw_c)

        # 3. Generate NEI (fabricated event) claims
        for rep in range(26):
            for reg in REGISTERS:
                seed = 5000 * f_idx + 200 * rep + len(candidates)
                raw_c = perturb_fabricated_event(fact, seed=seed)
                text = realize_surface_registers(raw_c.claim_text, reg)
                raw_c.claim_id = f"{raw_c.claim_id}_{reg}"
                raw_c.claim_text = text
                raw_c.register = reg
                candidates.append(raw_c)

    return candidates


def generate_and_freeze_benchmark(output_dir: str = "data/synthetic/ares_synth_v1") -> Tuple[pd.DataFrame, str]:
    """Build, validate, and write the frozen ARES-Synth v1 benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("ares_v2/data_manifests", exist_ok=True)

    facts = get_grounded_facts()
    fact_map = {f.fact_id: f for f in facts}

    candidates = build_candidate_pool(facts, target_per_fact=80)
    passed_claims, val_stats = run_validation_pipeline(candidates, fact_map)

    # Convert to DataFrame
    rows = []
    for c in passed_claims:
        rows.append({
            "item_id": c.claim_id,
            "scenario": c.scenario_id,
            "label": c.label,
            "perturbation_type": c.perturbation_type,
            "register": c.register,
            "fact_id": c.fact_id,
            "text": c.claim_text,
            "original_slot": c.original_slot,
            "perturbed_slot": c.perturbed_slot,
            "location_name": c.location_name,
            "geonames_id": c.geonames_id,
            "latitude": c.latitude,
            "longitude": c.longitude,
            "contradicting_sent_id": c.contradicting_sent_id or "",
            "split": "test",  # Test-only benchmark
            "data_kind": "synthetic_grounded",
        })

    df = pd.DataFrame(rows)
    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save parquet and csv
    parquet_path = os.path.join(output_dir, "dataset.parquet")
    csv_manifest_path = "ares_v2/data_manifests/ares_synth_v1.csv"
    df.to_parquet(parquet_path, index=False)

    # Manifest format
    manifest_df = df[[
        "item_id", "scenario", "label", "perturbation_type", "register",
        "fact_id", "geonames_id", "split"
    ]].copy()
    manifest_df.to_csv(csv_manifest_path, index=False)

    # Calculate sha256 of the parquet file
    with open(parquet_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    meta = {
        "benchmark_name": "ARES-Synth v1",
        "tag": "ares-synth-v1",
        "status": "SIMULATED — grounded synthetic",
        "n_total": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
        "scenario_counts": df["scenario"].value_counts().to_dict(),
        "perturbation_counts": df["perturbation_type"].value_counts().to_dict(),
        "register_counts": df["register"].value_counts().to_dict(),
        "sha256": file_hash,
        "validation_stats": val_stats,
    }

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return df, file_hash


if __name__ == "__main__":
    df, h = generate_and_freeze_benchmark()
    print(f"Generated ARES-Synth v1: n={len(df)}, sha256={h[:16]}...")
    print("Label breakdown:", df["label"].value_counts().to_dict())
