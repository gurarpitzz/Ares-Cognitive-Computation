"""LIAR dataset canonical ingestion and manifest generation."""

import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, List
import pandas as pd
import requests

from ares.common.paths import ARES_DATA_ROOT, ARES_ROOT
from ares.data.schema import save_canonical_parquet


RAW_URL = "https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip"
TSV_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

# Pre-registered binary collapse: TRUE (1) vs FALSE (0)
PRIMARY_BINARY_MAP = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0,
}


def download_and_extract(raw_dir: Path) -> None:
    """Download LIAR zip and extract canonical TSVs."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "liar_dataset.zip"
    retrieval_meta = raw_dir / "RETRIEVAL.json"

    if not zip_path.exists():
        print(f"Downloading LIAR dataset from {RAW_URL}...")
        resp = requests.get(RAW_URL, timeout=60)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        sha256 = hashlib.sha256(resp.content).hexdigest()

        retrieval_info = {
            "source_url": RAW_URL,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sha256": sha256,
            "bytes": len(resp.content),
        }
        retrieval_meta.write_text(json.dumps(retrieval_info, indent=2))
        print(f"LIAR zip downloaded ({len(resp.content)} bytes, sha256: {sha256[:16]}...)")

    # Extract TSV files
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    print(f"Extracted files to {raw_dir}")


def ingest_liar() -> pd.DataFrame:
    """Ingest LIAR into canonical schema and save Parquet."""
    raw_dir = ARES_DATA_ROOT / "raw" / "liar"
    download_and_extract(raw_dir)

    split_files = {
        "train": raw_dir / "train.tsv",
        "val": raw_dir / "valid.tsv",
        "test": raw_dir / "test.tsv",
    }

    records: List[Dict] = []

    for split_name, tsv_path in split_files.items():
        if not tsv_path.exists():
            raise FileNotFoundError(f"Missing LIAR split file: {tsv_path}")

        df_raw = pd.read_csv(tsv_path, sep="\t", header=None, names=TSV_COLUMNS, quoting=3)
        print(f"Processing LIAR {split_name}: {len(df_raw)} rows")

        for _, row in df_raw.iterrows():
            item_raw_id = str(row["id"])
            stmt = str(row["statement"]).strip() if pd.notna(row["statement"]) else ""
            label_raw = str(row["label"]).strip().lower()

            canonical_label = PRIMARY_BINARY_MAP.get(label_raw, -1)
            content_hash = hashlib.sha256(stmt.encode("utf-8")).hexdigest()

            meta = {
                "original_id": item_raw_id,
                "subject": str(row["subject"]) if pd.notna(row["subject"]) else "",
                "speaker": str(row["speaker"]) if pd.notna(row["speaker"]) else "",
                "job": str(row["speaker_job_title"]) if pd.notna(row["speaker_job_title"]) else "",
                "state": str(row["state_info"]) if pd.notna(row["state_info"]) else "",
                "party": str(row["party_affiliation"]) if pd.notna(row["party_affiliation"]) else "",
                "context": str(row["context"]) if pd.notna(row["context"]) else "",
                "counts": {
                    "barely_true": int(row["barely_true_counts"]) if pd.notna(row["barely_true_counts"]) else 0,
                    "false": int(row["false_counts"]) if pd.notna(row["false_counts"]) else 0,
                    "half_true": int(row["half_true_counts"]) if pd.notna(row["half_true_counts"]) else 0,
                    "mostly_true": int(row["mostly_true_counts"]) if pd.notna(row["mostly_true_counts"]) else 0,
                    "pants_on_fire": int(row["pants_on_fire_counts"]) if pd.notna(row["pants_on_fire_counts"]) else 0,
                },
                "content_sha256": content_hash,
            }

            records.append({
                "item_id": f"liar_{split_name}_{item_raw_id}",
                "split": split_name,
                "label_raw": label_raw,
                "label": canonical_label,
                "text": stmt,
                "lang": "en",
                "date": None,
                "group_id": str(row["speaker"]) if pd.notna(row["speaker"]) else None,
                "media_path": None,
                "meta": meta,
            })

    canonical_df = pd.DataFrame(records)
    out_dir = ARES_DATA_ROOT / "processed" / "liar"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "dataset.parquet"
    save_canonical_parquet(canonical_df, out_parquet)
    print(f"Saved canonical LIAR Parquet to {out_parquet} ({len(canonical_df)} rows)")

    # Generate split manifest
    manifest_records = []
    for _, row in canonical_df.iterrows():
        manifest_records.append({
            "item_id": row["item_id"],
            "split": row["split"],
            "label": row["label"],
            "content_sha256": row["meta"]["content_sha256"] if isinstance(row["meta"], dict) else json.loads(row["meta"])["content_sha256"],
        })
    manifest_df = pd.DataFrame(manifest_records)
    manifest_dir = ARES_ROOT / "ares_v2" / "data_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "liar.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Generated manifest: {manifest_path} ({len(manifest_df)} items)")

    return canonical_df


if __name__ == "__main__":
    df = ingest_liar()
    print("Split counts:\n", df["split"].value_counts())
    print("Label counts:\n", df["label_raw"].value_counts())
