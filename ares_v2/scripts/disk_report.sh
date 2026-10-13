#!/usr/bin/env bash
# Disk usage monitor and storage budget check (Target <= 150 GB on D:)
set -euo pipefail

ARES_ROOT="${ARES_ROOT:-$(pwd)}"

echo "=== ARES Disk Usage Report ==="
echo "Date: $(date)"
echo ""
echo "Directory sizes:"
for d in data cache milestones ares_v2; do
    if [ -d "$ARES_ROOT/$d" ]; then
        du -sh "$ARES_ROOT/$d" 2>/dev/null || true
    fi
done

echo ""
echo "Filesystem status:"
df -h "$ARES_ROOT"
