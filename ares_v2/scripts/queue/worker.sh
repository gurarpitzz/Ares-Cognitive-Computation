#!/usr/bin/env bash
# Dedicated single-worker queue runner for GPU jobs under tmux
set -euo pipefail

ARES_ROOT="${ARES_ROOT:-$(pwd)}"
QUEUE_DIR="$ARES_ROOT/run_queue"
LOG_DIR="$ARES_ROOT/logs"

mkdir -p "$QUEUE_DIR/pending" "$QUEUE_DIR/running" "$QUEUE_DIR/done" "$QUEUE_DIR/failed" "$LOG_DIR"

echo "[$(date)] ARES GPU Worker started. Listening on $QUEUE_DIR/pending..."

while true; do
    JOB=$(ls -1tr "$QUEUE_DIR/pending"/*.job 2>/dev/null | head -n 1 || true)
    if [ -n "$JOB" ] && [ -f "$JOB" ]; then
        JOB_NAME=$(basename "$JOB")
        RUNNING_JOB="$QUEUE_DIR/running/$JOB_NAME"
        mv "$JOB" "$RUNNING_JOB"
        
        # Source job details
        source "$RUNNING_JOB"
        
        LOG_FILE="$LOG_DIR/${EXP_ID}.log"
        TELEMETRY_CSV="$LOG_DIR/gpu_telemetry_${EXP_ID}.csv"
        
        echo "[$(date)] Starting Job: $EXP_ID"
        echo "[$(date)] Command: $CMD"
        
        # Start GPU telemetry in background
        "$ARES_ROOT/ares_v2/scripts/queue/gpu_telemetry.sh" "$EXP_ID" "$TELEMETRY_CSV" &
        TELEM_PID=$!
        
        START_TIME=$(date +%s)
        EXIT_CODE=0
        eval "$CMD" > "$LOG_FILE" 2>&1 || EXIT_CODE=$?
        END_TIME=$(date +%s)
        WALL_CLOCK=$((END_TIME - START_TIME))
        
        # Stop telemetry
        kill "$TELEM_PID" 2>/dev/null || true
        wait "$TELEM_PID" 2>/dev/null || true
        
        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "[$(date)] Job $EXP_ID finished successfully (${WALL_CLOCK}s)"
            mv "$RUNNING_JOB" "$QUEUE_DIR/done/$JOB_NAME"
        else
            echo "[$(date)] Job $EXP_ID FAILED with code $EXIT_CODE (${WALL_CLOCK}s)"
            mv "$RUNNING_JOB" "$QUEUE_DIR/failed/$JOB_NAME"
        fi
    else
        sleep 5
    fi
done
