#!/usr/bin/env bash
# GPU Telemetry logger running every 30 seconds
set -euo pipefail

EXP_ID="${1:-unknown}"
OUTPUT_CSV="${2:-logs/gpu_telemetry_${EXP_ID}.csv}"
mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "Logging GPU telemetry for $EXP_ID to $OUTPUT_CSV"
exec nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 30 > "$OUTPUT_CSV"
