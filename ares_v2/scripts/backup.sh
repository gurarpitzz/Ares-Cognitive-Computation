#!/usr/bin/env bash
# Nightly / milestone backup script to secondary storage (E:\ares_backup)
set -euo pipefail

ARES_ROOT="${ARES_ROOT:-$(pwd)}"
BACKUP_DIR="${ARES_BACKUP_DIR:-/mnt/e/ares_backup}"

mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting backup from $ARES_ROOT to $BACKUP_DIR..."

rsync -av --delete \
    --exclude="data/raw" \
    --exclude="cache" \
    --exclude=".venv" \
    --exclude="*.tmp" \
    --exclude="__pycache__" \
    "$ARES_ROOT/milestones" \
    "$ARES_ROOT/plan" \
    "$ARES_ROOT/ares_v2" \
    "$ARES_ROOT/docs" \
    "$BACKUP_DIR/"

echo "[$(date)] Backup completed successfully."
