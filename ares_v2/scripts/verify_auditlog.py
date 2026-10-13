"""
CLI Utility to Verify Cryptographic Audit Log Integrity (Milestone M14, Claim C084).

Usage:
    python ares_v2/scripts/verify_auditlog.py [path_to_audit_log.jsonl]
"""

from __future__ import annotations

import sys
from pathlib import Path

from ares.service.auditlog import verify_auditlog


def main() -> None:
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = Path("milestones/M14_service_ops/data/audit_log.jsonl")

    if not log_path.exists():
        print(f"Error: Audit log file not found at {log_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying cryptographic hash chain for {log_path}...")
    res = verify_auditlog(log_path)

    if res.is_valid:
        print(f"SUCCESS: Audit log is valid! Verified {res.total_blocks} blocks.")
        print(f"Current Head Hash: {res.head_hash}")
        sys.exit(0)
    else:
        print(f"FAILED: Tamper detected at block {res.corrupted_block_index}!", file=sys.stderr)
        print(f"Reason: {res.error_reason}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
