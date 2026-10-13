"""Integrity Test: Enforce Protocol §4 Results Schema.

Validates all JSON files in milestones/*/results/ against ExperimentResult schema.
"""

import json
from pathlib import Path
from ares.common.paths import MILESTONES_ROOT
from ares.common.results import ExperimentResult


def test_all_results_schema():
    results_files = list(MILESTONES_ROOT.rglob("results/*.json"))
    errors = []
    
    for rf in results_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)
            ExperimentResult.model_validate(data)
        except Exception as e:
            errors.append(f"{rf.relative_to(MILESTONES_ROOT)}: validation failed: {e}")
            
    assert len(errors) == 0, (
        f"Found {len(errors)} invalid result JSON files:\n" + "\n".join(errors)
    )
