"""Integrity Test: Enforce Blind Configs & No Target Chasing.

Scans src/, configs/, scripts/ for distinctive manuscript result values.
Fails if any forbidden number appears as a literal unless accompanied by
'# ledger:CXXX hyperparameter'.
"""

import re
from pathlib import Path
import yaml
from ares.common.paths import ARES_ROOT

ALLOW_COMMENT_PATTERN = re.compile(r"#\s*ledger:C\d+\s+hyperparameter", re.IGNORECASE)


def load_forbidden_values() -> list[str]:
    yaml_path = Path(__file__).parent / "manuscript_values.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [str(v) for v in data.get("forbidden_result_values", [])]


def scan_file_for_numbers(file_path: Path, forbidden: list[str]) -> list[tuple[int, str, str]]:
    violations = []
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return violations
        
    lines = content.splitlines()
    for line_idx, line in enumerate(lines, start=1):
        # Check if line has allow comment
        if ALLOW_COMMENT_PATTERN.search(line):
            continue
            
        for val in forbidden:
            # Word boundary / numeric token matching
            pattern = rf"(?<![\w.]){re.escape(val)}(?![\w.])"
            if re.search(pattern, line):
                violations.append((line_idx, val, line.strip()))
                
    return violations


def test_no_manuscript_numbers():
    forbidden = load_forbidden_values()
    assert len(forbidden) > 0, "No forbidden numbers loaded."
    
    code_dirs = [
        ARES_ROOT / "ares_v2" / "src",
        ARES_ROOT / "ares_v2" / "configs",
        ARES_ROOT / "ares_v2" / "scripts",
    ]
    
    all_violations = []
    for c_dir in code_dirs:
        if not c_dir.exists():
            continue
        for ext in ["*.py", "*.yaml", "*.yml", "*.sh", "*.json"]:
            for f_path in c_dir.rglob(ext):
                # Skip legacy folder
                if "legacy" in f_path.parts:
                    continue
                violations = scan_file_for_numbers(f_path, forbidden)
                for line_idx, val, line_text in violations:
                    all_violations.append(
                        f"{f_path.relative_to(ARES_ROOT)}:{line_idx}: forbidden value '{val}' found: {line_text}"
                    )
                    
    assert len(all_violations) == 0, (
        f"Found {len(all_violations)} integrity violations (manuscript numbers in code/configs):\n"
        + "\n".join(all_violations)
    )
