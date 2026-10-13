"""Claims ledger management CLI tool for ARES rebuild."""

import argparse
import datetime
import shutil
import sys
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "claim_id",
    "section",
    "location",
    "claim_text",
    "reported_value",
    "claim_type",
    "reproducible_by_code",
    "target_milestone",
    "status",
    "measured_value",
    "ci95",
    "evidence_path",
    "manuscript_action",
    "notes",
]

ALLOWED_STATUSES = {
    "UNVERIFIED",
    "REPRODUCED",
    "REVISED",
    "CORRECTED-THEORY",
    "REFRAMED",
    "REMOVE",
    "BLOCKED",
}


def get_ledger_path(args_path: str = None) -> Path:
    if args_path:
        return Path(args_path)
    # Default to plan/claims_ledger.csv relative to repo root
    current = Path(__file__).resolve().parent.parent.parent
    return current / "plan" / "claims_ledger.csv"


def cmd_validate(args):
    path = get_ledger_path(args.path)
    if not path.exists():
        print(f"ERROR: Ledger file not found at {path}", file=sys.stderr)
        sys.exit(1)
        
    df = pd.read_csv(path)
    errors = []
    
    # Check columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
        
    # Check unique claim_id
    if "claim_id" in df.columns:
        dupes = df[df["claim_id"].duplicated()]["claim_id"].tolist()
        if dupes:
            errors.append(f"Duplicate claim_id values found: {dupes}")
            
    # Check statuses
    if "status" in df.columns:
        invalid_statuses = df[~df["status"].isin(ALLOWED_STATUSES)]["status"].dropna().unique()
        if len(invalid_statuses) > 0:
            errors.append(f"Invalid status values found: {list(invalid_statuses)}")
            
    if errors:
        print("FAIL: Ledger validation failed with errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"PASS: Ledger {path.name} is valid. Total claims: {len(df)}")


def cmd_summary(args):
    path = get_ledger_path(args.path)
    df = pd.read_csv(path)
    
    print("\n# Claims Ledger Summary\n")
    print(f"**Total Claims:** {len(df)}\n")
    
    status_counts = df["status"].value_counts()
    print("### Status Breakdown")
    print("| Status | Count | Percentage |")
    print("| :--- | :--- | :--- |")
    for status, count in status_counts.items():
        pct = (count / len(df)) * 100.0
        print(f"| `{status}` | {count} | {pct:.1f}% |")
        
    print("\n### Target Milestone Breakdown")
    ms_counts = df["target_milestone"].value_counts()
    print("| Milestone | Count |")
    print("| :--- | :--- |")
    for ms, count in ms_counts.items():
        print(f"| `{ms}` | {count} |")
    print()


def cmd_apply_delta(args):
    ledger_path = get_ledger_path(args.path)
    milestone = args.milestone
    delta_path = ledger_path.parent.parent / "milestones" / milestone / "ledger_delta.csv"
    
    if not delta_path.exists():
        print(f"ERROR: Delta file not found at {delta_path}", file=sys.stderr)
        sys.exit(1)
        
    delta_df = pd.read_csv(delta_path, dtype=str)
    main_df = pd.read_csv(ledger_path, dtype=str)
    
    # Create history snapshot before applying delta
    history_dir = ledger_path.parent / "ledger_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    snapshot_path = history_dir / f"{today}_{milestone}.csv"
    shutil.copy(ledger_path, snapshot_path)
    print(f"Saved snapshot to {snapshot_path}")
    
    # Merge delta into main_df by claim_id
    updated_count = 0
    for _, row in delta_df.iterrows():
        cid = row["claim_id"]
        mask = main_df["claim_id"] == cid
        if mask.any():
            for col in delta_df.columns:
                if pd.notna(row[col]):
                    main_df.loc[mask, col] = row[col]
            updated_count += 1
        else:
            print(f"WARNING: Claim ID {cid} in delta not found in main ledger.")
            
    main_df.to_csv(ledger_path, index=False)
    print(f"Successfully applied delta for {milestone}. Updated {updated_count} claims.")


def cmd_check_evidence(args):
    ledger_path = get_ledger_path(args.path)
    df = pd.read_csv(ledger_path)
    repo_root = ledger_path.parent.parent
    
    non_unverified = df[df["status"] != "UNVERIFIED"]
    missing = []
    
    for _, row in non_unverified.iterrows():
        ep = row.get("evidence_path")
        if pd.isna(ep) or not ep:
            missing.append(f"{row['claim_id']} ({row['status']}): missing evidence_path")
        else:
            p = repo_root / str(ep)
            if not p.exists():
                missing.append(f"{row['claim_id']}: evidence file does not exist: {ep}")
                
    if missing:
        print(f"FAIL: {len(missing)} claims missing evidence files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"PASS: All {len(non_unverified)} verified claims have valid evidence paths.")


def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--path", type=str, default=None, help="Path to claims_ledger.csv")
    
    parser = argparse.ArgumentParser(description="ARES Claims Ledger Tooling", parents=[parent_parser])
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("validate", parents=[parent_parser], help="Validate ledger schema and constraints")
    subparsers.add_parser("summary", parents=[parent_parser], help="Print summary of ledger statuses")
    
    delta_parser = subparsers.add_parser("apply-delta", parents=[parent_parser], help="Apply milestone delta to ledger")
    delta_parser.add_argument("milestone", type=str, help="Milestone folder name (e.g. M00_foundation)")
    
    subparsers.add_parser("check-evidence", parents=[parent_parser], help="Check evidence paths exist")
    
    args = parser.parse_args()
    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "apply-delta":
        cmd_apply_delta(args)
    elif args.command == "check-evidence":
        cmd_check_evidence(args)


if __name__ == "__main__":
    main()
