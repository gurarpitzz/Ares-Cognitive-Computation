"""Resumable training loop wrapper with atomic checkpointing and full RNG restoration."""

import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import torch


class CheckpointManager:
    """Manages atomic checkpointing for resumable training."""

    def __init__(self, checkpoint_dir: Path, checkpoint_interval_steps: int = 500, checkpoint_interval_seconds: float = 1800.0):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval_steps = checkpoint_interval_steps
        self.interval_seconds = checkpoint_interval_seconds
        self.last_checkpoint_time = time.time()

    def should_checkpoint(self, step: int) -> bool:
        if step > 0 and step % self.interval_steps == 0:
            return True
        if (time.time() - self.last_checkpoint_time) >= self.interval_seconds:
            return True
        return False

    def save(
        self,
        step: int,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Atomically save checkpoint to disk."""
        target_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        tmp_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt.tmp"

        state = {
            "step": step,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict() if scaler else None,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "extra_state": extra_state or {},
        }

        torch.save(state, tmp_path)
        # Atomic rename
        if tmp_path.exists():
            os.replace(tmp_path, target_path)

        # Update latest symlink/marker
        latest_marker = self.checkpoint_dir / "latest_checkpoint.txt"
        latest_marker.write_text(str(target_path.name), encoding="utf-8")
        self.last_checkpoint_time = time.time()
        return target_path

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Tuple[int, int, Dict[str, Any]]:
        """Load the most recent valid checkpoint."""
        latest_marker = self.checkpoint_dir / "latest_checkpoint.txt"
        if not latest_marker.exists():
            raise FileNotFoundError("No latest_checkpoint marker found.")

        filename = latest_marker.read_text(encoding="utf-8").strip()
        ckpt_path = self.checkpoint_dir / filename
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        if scaler and state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])

        # Restore RNG states
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

        return state["step"], state["epoch"], state.get("extra_state", {})
