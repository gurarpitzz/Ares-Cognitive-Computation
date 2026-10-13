#!/usr/bin/env bash
# Master runner for ARES Smoke Test Suite (<15 min execution)
set -euo pipefail

ARES_ROOT="${ARES_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export PYTHONPATH="$ARES_ROOT/ares_v2/src:$PYTHONPATH"

echo "=========================================="
echo "Starting ARES Foundation Smoke Test Suite"
echo "=========================================="

echo "[1/3] Running Smoke Test (a): Seeded Matmul & Determinism..."
python "$ARES_ROOT/ares_v2/scripts/smoke/smoke_determinism.py"

echo "[2/3] Running Smoke Test (b): Kill-and-Resume State Check..."
python "$ARES_ROOT/ares_v2/scripts/smoke/smoke_kill_resume.py"

echo "[3/3] Running Smoke Test (c): VRAM Capacity & Throughput Suite..."
python "$ARES_ROOT/ares_v2/scripts/smoke/smoke_vram_fit.py"

echo "=========================================="
echo "All M00 Smoke Tests Completed Successfully!"
echo "=========================================="
