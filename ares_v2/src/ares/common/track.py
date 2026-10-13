"""Thin experiment tracking interface (canonical storage remains results JSON)."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from ares.common.paths import MILESTONES_ROOT


class RunTracker:
    """Local file-backed experiment tracker."""

    def __init__(self, experiment_id: str, milestone: str, tracking_dir: Optional[Path] = None):
        self.experiment_id = experiment_id
        self.milestone = milestone
        self.tracking_dir = tracking_dir or (MILESTONES_ROOT / milestone / "logs")
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.tracking_dir / f"{experiment_id}_track.jsonl"
        self.params: Dict[str, Any] = {}

    def log_params(self, params: Dict[str, Any]) -> None:
        self.params.update(params)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "params", "data": params}) + "\n")

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "metrics", "step": step, "data": metrics}) + "\n")

    def finish(self) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "status", "data": "FINISHED"}) + "\n")


def init_tracker(experiment_id: str, milestone: str) -> RunTracker:
    return RunTracker(experiment_id, milestone)
