"""Data manifest and cryptographic integrity utilities."""

import hashlib
from pathlib import Path
from typing import List, Optional
import pandas as pd


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(dir_path: Path) -> str:
    """Compute deterministic SHA-256 digest of an entire directory based on sorted file hashes."""
    h = hashlib.sha256()
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            rel_path = p.relative_to(dir_path).as_posix()
            h.update(rel_path.encode("utf-8"))
            h.update(sha256_file(p).encode("utf-8"))
    return h.hexdigest()


def build_manifest(df: pd.DataFrame, id_col: str, split_col: str, label_col: Optional[str] = None) -> pd.DataFrame:
    """Standardize and validate a dataset manifest.
    
    Required columns: id, split. Optional: label, sha256.
    """
    manifest = pd.DataFrame()
    manifest["item_id"] = df[id_col].astype(str)
    manifest["split"] = df[split_col].astype(str)
    if label_col and label_col in df.columns:
        manifest["label"] = df[label_col].astype(str)
    
    # Sort deterministically
    manifest = manifest.sort_values(by=["split", "item_id"]).reset_index(drop=True)
    return manifest


def manifest_hash(manifest_df: pd.DataFrame) -> str:
    """Compute canonical sha256 digest of a manifest DataFrame."""
    canonical_csv = manifest_df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical_csv).hexdigest()
