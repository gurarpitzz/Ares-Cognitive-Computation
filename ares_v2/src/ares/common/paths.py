"""Path definitions and resolution for ARES."""

import os
from pathlib import Path

# Base repository root: default to 5 levels up from this file (d:\ares)
_DEFAULT_ROOT = Path(__file__).resolve().parents[4]

ARES_ROOT = Path(os.environ.get("ARES_ROOT", str(_DEFAULT_ROOT))).resolve()
ARES_DATA_ROOT = Path(os.environ.get("ARES_DATA_ROOT", str(ARES_ROOT / "data"))).resolve()
ARES_CACHE_ROOT = Path(os.environ.get("ARES_CACHE_ROOT", str(ARES_ROOT / "cache"))).resolve()
MILESTONES_ROOT = Path(os.environ.get("ARES_MILESTONES_ROOT", str(ARES_ROOT / "milestones"))).resolve()
CONFIGS_ROOT = Path(os.environ.get("ARES_CONFIGS_ROOT", str(ARES_ROOT / "ares_v2" / "configs"))).resolve()
SCRIPTS_ROOT = Path(os.environ.get("ARES_SCRIPTS_ROOT", str(ARES_ROOT / "ares_v2" / "scripts"))).resolve()
PLAN_ROOT = Path(os.environ.get("ARES_PLAN_ROOT", str(ARES_ROOT / "plan"))).resolve()
CLAIMS_LEDGER_PATH = PLAN_ROOT / "claims_ledger.csv"
TEST_SET_ACCESS_LOG = MILESTONES_ROOT / "test_set_access_log.csv"
COMPUTE_LEDGER_PATH = MILESTONES_ROOT / "compute_ledger.csv"


def ensure_dirs() -> None:
    """Ensure standard runtime directories exist."""
    ARES_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    ARES_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    MILESTONES_ROOT.mkdir(parents=True, exist_ok=True)
