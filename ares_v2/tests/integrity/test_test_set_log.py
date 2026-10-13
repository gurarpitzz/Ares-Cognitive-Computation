"""Integrity Test: Enforce Protocol §1 Rule 5 Test Set Access Logging.

Validates that every results JSON with split=='test' has a matching entry
in milestones/test_set_access_log.csv.
"""

import json
import pandas as pd
from ares.common.paths import MILESTONES_ROOT, TEST_SET_ACCESS_LOG


def test_test_set_access_logging():
    if not TEST_SET_ACCESS_LOG.exists():
        log_df = pd.DataFrame(columns=["timestamp", "milestone", "experiment_id", "split", "dataset", "git_commit", "reason"])
    else:
        log_df = pd.read_csv(TEST_SET_ACCESS_LOG)
        
    logged_exp_ids = set(log_df["experiment_id"].dropna().astype(str)) if not log_df.empty else set()
    
    results_files = list(MILESTONES_ROOT.rglob("results/*.json"))
    missing_logs = []
    
    for rf in results_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("split") == "test":
                exp_id = data.get("experiment_id", "")
                if exp_id not in logged_exp_ids:
                    missing_logs.append(
                        f"{rf.name} evaluated on 'test' split but not logged in test_set_access_log.csv"
                    )
        except Exception:
            continue
            
    assert len(missing_logs) == 0, (
        f"Found {len(missing_logs)} unlogged test set evaluations:\n" + "\n".join(missing_logs)
    )
