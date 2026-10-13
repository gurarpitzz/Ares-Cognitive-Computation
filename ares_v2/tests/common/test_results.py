"""Unit tests for results schema validation and serialization."""

from ares.common.results import ExperimentResult, write_result


def test_results_serialization(tmp_path):
    out_file = tmp_path / "test_result.json"
    dummy_data = {
        "experiment_id": "M00-test-run",
        "milestone": "M00",
        "ledger_ids": ["C097"],
        "data_kind": "real",
        "git_commit": "abc1234",
        "dirty_worktree": False,
        "config_path": "configs/test.yaml",
        "config_hash": "sha256:1111",
        "data_manifest_hash": "sha256:2222",
        "split": "test",
        "seed": 42,
        "started_at": "2026-10-13T10:00:00Z",
        "finished_at": "2026-10-13T10:05:00Z",
        "wall_clock_s": 300.0,
        "n_examples": 100,
        "metrics": {"macro_f1": 0.85, "accuracy": 0.88},
        "ci95": {"macro_f1": [0.80, 0.90]},
    }
    
    res = write_result(dummy_data, out_file, allow_dirty=True)
    assert isinstance(res, ExperimentResult)
    assert out_file.exists()
    assert res.experiment_id == "M00-test-run"
