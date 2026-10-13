#!/usr/bin/env bash
# Check status of the ARES job queue
set -euo pipefail

QUEUE_DIR="${ARES_ROOT:-$(pwd)}/run_queue"

echo "=== ARES Job Queue Status ==="
echo "Pending Jobs : $(ls -1 "$QUEUE_DIR/pending"/*.job 2>/dev/null | wc -l)"
echo "Running Jobs : $(ls -1 "$QUEUE_DIR/running"/*.job 2>/dev/null | wc -l)"
echo "Done Jobs    : $(ls -1 "$QUEUE_DIR/done"/*.job 2>/dev/null | wc -l)"
echo "Failed Jobs  : $(ls -1 "$QUEUE_DIR/failed"/*.job 2>/dev/null | wc -l)"
echo ""
if [ "$(ls -1 "$QUEUE_DIR/running"/*.job 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Currently running:"
    for j in "$QUEUE_DIR/running"/*.job; do
        source "$j"
        echo " - $EXP_ID: $CMD"
    done
fi
