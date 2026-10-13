"""Results schema and serialization for ARES experiments."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class HardwareInfo(BaseModel):
    gpu: str = "NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)"
    cpu: str = "13th Gen Intel Core i7-13620H"
    ram_gb: int = 16
    wsl_mem_limit_gb: int = 12
    os: str = "Windows 11 24H2 + WSL2 Ubuntu 26.04"


class SoftwareInfo(BaseModel):
    python: str = "3.11.17"
    torch: str = ""
    transformers: str = ""
    cuda: str = ""


class ExperimentResult(BaseModel):
    experiment_id: str
    milestone: str
    ledger_ids: List[str]
    data_kind: Literal["real", "simulated", "synthetic_grounded"]
    git_commit: str
    dirty_worktree: bool
    config_path: str
    config_hash: str
    data_manifest_hash: str
    split: Literal["train", "val", "test"]
    seed: int
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    precision: str = "bf16"
    micro_batch: int = 8
    grad_accum: int = 4
    effective_batch: int = 32
    resumed_from_checkpoint: bool = False
    gpu_telemetry_path: Optional[str] = None
    thermal_throttle_detected: bool = False
    software: SoftwareInfo = Field(default_factory=SoftwareInfo)
    started_at: str
    finished_at: str
    wall_clock_s: float
    n_examples: int
    metrics: Dict[str, float]
    ci95: Dict[str, List[float]] = Field(default_factory=dict)
    per_class: Optional[Dict[str, Dict[str, float]]] = None
    predictions_path: Optional[str] = None
    notes: str = ""


def get_git_info() -> tuple[str, bool]:
    """Retrieve the current HEAD commit hash and worktree dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        is_dirty = len(status) > 0
        return commit, is_dirty
    except Exception:
        return "unknown", True


def get_software_info() -> SoftwareInfo:
    """Retrieve runtime software versions."""
    torch_ver = ""
    cuda_ver = ""
    transformers_ver = ""
    try:
        import torch
        torch_ver = torch.__version__
        cuda_ver = torch.version.cuda or "none"
    except ImportError:
        torch_ver = "not_installed"
        cuda_ver = "none"
        
    try:
        import transformers
        transformers_ver = transformers.__version__
    except ImportError:
        transformers_ver = "not_installed"
        
    return SoftwareInfo(
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        torch=torch_ver,
        transformers=transformers_ver,
        cuda=cuda_ver
    )


def write_result(
    result_data: Dict[str, Any],
    output_path: Path,
    allow_dirty: bool = False
) -> ExperimentResult:
    """Validate and write experiment results JSON with provenance enforcement.
    
    Args:
        result_data: Dictionary of experiment outcome data.
        output_path: Target JSON file path.
        allow_dirty: If False, raises RuntimeError if git worktree is dirty.
        
    Returns:
        Validated ExperimentResult instance.
    """
    commit, is_dirty = get_git_info()
    if is_dirty and not allow_dirty:
        raise RuntimeError(
            "Protocol violation: refusing to write results with dirty git worktree. "
            "Commit your changes or pass allow_dirty=True for debugging."
        )
    
    # Auto-fill provenance fields if missing
    if "git_commit" not in result_data or not result_data["git_commit"]:
        result_data["git_commit"] = commit
    if "dirty_worktree" not in result_data:
        result_data["dirty_worktree"] = is_dirty
    if "software" not in result_data:
        result_data["software"] = get_software_info().model_dump()
        
    # Validate with Pydantic
    model = ExperimentResult(**result_data)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(model.model_dump_json(indent=2))
        
    return model
