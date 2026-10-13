"""Deterministic seeding utilities for reproducible execution."""

import os
import random
import numpy as np
import torch


def set_all(seed: int) -> None:
    """Set random seeds across Python, NumPy, PyTorch (CPU and CUDA) and enforce determinism.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms where supported
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """Worker initialization function for PyTorch DataLoader to ensure worker-level determinism.
    
    Args:
        worker_id: The ID of the dataloader worker.
        base_seed: Master seed for the run.
    """
    worker_seed = (base_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
