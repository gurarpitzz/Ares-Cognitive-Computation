#!/usr/bin/env bash
# Submit a job command to the ARES FIFO queue
set -euo pipefail

QUEUE_DIR="${ARES_ROOT:-$(pwd)}/run_queue"
mkdir -p "$QUEUE_DIR/pending" "$QUEUE_DIR/running" "$QUEUE_DIR/done" "$QUEUE_DIR/failed"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment_id> <command...>"
    exit 1
fi

EXP_ID="$1"
shift
CMD="$*"

JOB_FILE="$QUEUE_DIR/pending/$(date +%s%N)_${EXP_ID}.job"

cat <<EOF > "$JOB_FILE"
EXP_ID="$EXP_ID"
CMD="$CMD"
SUBMITTED_AT="$(date -Iseconds)"
EOF

echo "Queued job $EXP_ID -> $JOB_FILE"
