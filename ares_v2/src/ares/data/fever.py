"""FEVER 1.0 canonical ingestion and manifest generation."""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import requests

from ares.common.paths import ARES_DATA_ROOT, ARES_ROOT
from ares.data.schema import save_canonical_parquet


BASE_URL = "https://fever.ai/download/fever"
SPLIT_URLS = {
    "train": f"{BASE_URL}/train.jsonl",
    "val": f"{BASE_URL}/paper_dev.jsonl",
    "test": f"{BASE_URL}/paper_test.jsonl",
}

LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}


def download_fever_splits(raw_dir: Path) -> None:
    """Download FEVER split jsonl files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieval_meta = raw_dir / "RETRIEVAL.json"
    meta_records = {}

    for split_name, url in SPLIT_URLS.items():
        fname = Path(url).name
        target_path = raw_dir / fname
        if not target_path.exists():
            print(f"Downloading FEVER {split_name} from {url}...")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            # Compute hash
            sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
            meta_records[split_name] = {
                "source_url": url,
                "file_name": fname,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sha256": sha256,
                "bytes": target_path.stat().st_size,
            }
            print(f"Saved {fname} ({target_path.stat().st_size} bytes, sha256: {sha256[:16]}...)")

    if meta_records and not retrieval_meta.exists():
        retrieval_meta.write_text(json.dumps(meta_records, indent=2))


def ingest_fever() -> pd.DataFrame:
    """Ingest FEVER into canonical schema and save Parquet."""
    raw_dir = ARES_DATA_ROOT / "raw" / "fever"
    download_fever_splits(raw_dir)

    split_files = {
        "train": raw_dir / "train.jsonl",
        "val": raw_dir / "paper_dev.jsonl",
        "test": raw_dir / "paper_test.jsonl",
    }

    records: List[Dict] = []

    for split_name, jsonl_path in split_files.items():
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing FEVER split file: {jsonl_path}")

        print(f"Processing FEVER {split_name} from {jsonl_path}...")
        line_count = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                line_count += 1
                item = json.loads(line)

                claim_text = str(item.get("claim", "")).strip()
                raw_label = str(item.get("label", "")).strip()
                canonical_label = LABEL_MAP.get(raw_label, -1)
                item_raw_id = str(item.get("id", line_count))

                content_hash = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
                item_id = f"fever_{split_name}_{item_raw_id}"

                evidence = item.get("evidence", [])
                meta = {
                    "original_id": item_raw_id,
                    "verifiable": item.get("verifiable", ""),
                    "evidence": evidence,
                    "content_sha256": content_hash,
                }

                records.append({
                    "item_id": item_id,
                    "split": split_name,
                    "label_raw": raw_label,
                    "label": canonical_label,
                    "text": claim_text,
                    "lang": "en",
                    "date": None,
                    "group_id": None,
                    "media_path": None,
                    "meta": meta,
                })

        print(f"Loaded {line_count} records from {split_name}")

    canonical_df = pd.DataFrame(records)
    out_dir = ARES_DATA_ROOT / "processed" / "fever"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "dataset.parquet"
    save_canonical_parquet(canonical_df, out_parquet)
    print(f"Saved canonical FEVER Parquet to {out_parquet} ({len(canonical_df)} rows)")

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
    manifest_path = manifest_dir / "fever.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Generated manifest: {manifest_path} ({len(manifest_df)} items)")

    return canonical_df


if __name__ == "__main__":
    df = ingest_fever()
    print("Split counts:\n", df["split"].value_counts())
    print("Label counts:\n", df["label_raw"].value_counts())
