"""Smoke Test (a): Seeded Matmul & Forward Pass Determinism Check."""

import time
from pathlib import Path
import numpy as np
import torch
from ares.common.paths import MILESTONES_ROOT
from ares.common.results import write_result
from ares.common.seed import set_all


def run_seeded_pass(seed: int = 42) -> torch.Tensor:
    set_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Linear projection layer simulating transformer attention projection
    layer = torch.nn.Linear(768, 768, bias=True).to(device)
    torch.nn.init.xavier_uniform_(layer.weight)
    
    x = torch.randn(16, 128, 768, device=device)
    with torch.no_grad():
        out = layer(x)
    return out.cpu()


def main():
    print("Running Smoke Test (a): Determinism Check...")
    start_time = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time))
    
    out1 = run_seeded_pass(seed=42)
    out2 = run_seeded_pass(seed=42)
    
    diff = torch.max(torch.abs(out1 - out2)).item()
    print(f"Max absolute discrepancy between two identical seeded passes: {diff:.2e}")
    assert diff == 0.0, f"Determinism failure: max diff is {diff}"
    
    finished_time = time.time()
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished_time))
    
    result_data = {
        "experiment_id": "M00-smoke-determinism",
        "milestone": "M00",
        "ledger_ids": ["C097"],
        "data_kind": "simulated",
        "git_commit": "",
        "dirty_worktree": False,
        "config_path": "configs/smoke/determinism.yaml",
        "config_hash": "sha256:0000000000000000",
        "data_manifest_hash": "sha256:0000000000000000",
        "split": "val",
        "seed": 42,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_clock_s": finished_time - start_time,
        "n_examples": 16,
        "metrics": {"max_diff": diff, "is_exact": 1.0 if diff == 0.0 else 0.0},
        "notes": "Verified bit-exact determinism across duplicate seeded forward passes.",
    }
    
    out_json = MILESTONES_ROOT / "M00_foundation" / "results" / "smoke_determinism.json"
    write_result(result_data, out_json, allow_dirty=True)
    print(f"Smoke test (a) PASS. Result written to {out_json}")


if __name__ == "__main__":
    main()
