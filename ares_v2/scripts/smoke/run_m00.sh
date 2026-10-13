#!/usr/bin/env bash
# Complete M00 Milestone Reproduction Runner
set -euo pipefail

ARES_ROOT="${ARES_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export PYTHONPATH="$ARES_ROOT/ares_v2/src:$PYTHONPATH"

echo "=========================================="
echo "Executing M00 Foundation & Integrity Tests"
echo "=========================================="

echo "[1/4] Running pytest unit and integrity tests..."
pytest -v "$ARES_ROOT/ares_v2/tests"

echo "[2/4] Validating Claims Ledger..."
python "$ARES_ROOT/ares_v2/scripts/ledger.py" validate --path "$ARES_ROOT/plan/claims_ledger.csv"
python "$ARES_ROOT/ares_v2/scripts/ledger.py" summary --path "$ARES_ROOT/plan/claims_ledger.csv"

echo "[3/4] Running Smoke and VRAM Fit Suite..."
bash "$ARES_ROOT/ares_v2/scripts/smoke/run_smoke.sh"

echo "[4/4] Executing Backup to E:..."
bash "$ARES_ROOT/ares_v2/scripts/backup.sh"

echo "=========================================="
echo "M00 Foundation Execution Complete!"
echo "=========================================="
