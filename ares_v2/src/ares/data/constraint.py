"""Constraint@AAAI-2021 COVID fake news dataset canonical ingestion and manifest generation."""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import requests

from ares.common.paths import ARES_DATA_ROOT, ARES_ROOT
from ares.data.schema import save_canonical_parquet


BASE_URL = "https://raw.githubusercontent.com/diptamath/covid_fake_news/master/data"
SPLIT_URLS = {
    "train": f"{BASE_URL}/Constraint_Train.csv",
    "val": f"{BASE_URL}/Constraint_Val.csv",
    "test": f"{BASE_URL}/english_test_with_labels.csv",
}

LABEL_MAP = {
    "real": 1,
    "fake": 0,
}


def download_constraint_splits(raw_dir: Path) -> None:
    """Download Constraint-COVID split CSV files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieval_meta = raw_dir / "RETRIEVAL.json"
    meta_records = {}

    for split_name, url in SPLIT_URLS.items():
        fname = Path(url).name
        target_path = raw_dir / fname
        if not target_path.exists():
            print(f"Downloading Constraint {split_name} from {url}...")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            target_path.write_bytes(resp.content)
            sha256 = hashlib.sha256(resp.content).hexdigest()
            meta_records[split_name] = {
                "source_url": url,
                "file_name": fname,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sha256": sha256,
                "bytes": len(resp.content),
            }
            print(f"Saved {fname} ({len(resp.content)} bytes, sha256: {sha256[:16]}...)")

    if meta_records and not retrieval_meta.exists():
        retrieval_meta.write_text(json.dumps(meta_records, indent=2))


def ingest_constraint() -> pd.DataFrame:
    """Ingest Constraint dataset into canonical schema and save Parquet."""
    raw_dir = ARES_DATA_ROOT / "raw" / "constraint"
    download_constraint_splits(raw_dir)

    split_files = {
        "train": raw_dir / "Constraint_Train.csv",
        "val": raw_dir / "Constraint_Val.csv",
        "test": raw_dir / "english_test_with_labels.csv",
    }

    records: List[Dict] = []

    for split_name, csv_path in split_files.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing Constraint file: {csv_path}")

        df_raw = pd.read_csv(csv_path)
        print(f"Processing Constraint {split_name}: {len(df_raw)} rows")

        for _, row in df_raw.iterrows():
            item_raw_id = str(row["id"])
            tweet_text = str(row["tweet"]).strip() if pd.notna(row["tweet"]) else ""
            label_raw = str(row["label"]).strip().lower()

            canonical_label = LABEL_MAP.get(label_raw, -1)
            content_hash = hashlib.sha256(tweet_text.encode("utf-8")).hexdigest()

            meta = {
                "original_id": item_raw_id,
                "content_sha256": content_hash,
            }

            records.append({
                "item_id": f"constraint_{split_name}_{item_raw_id}",
                "split": split_name,
                "label_raw": label_raw,
                "label": canonical_label,
                "text": tweet_text,
                "lang": "en",
                "date": None,
                "group_id": None,
                "media_path": None,
                "meta": meta,
            })

    canonical_df = pd.DataFrame(records)
    out_dir = ARES_DATA_ROOT / "processed" / "constraint"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "dataset.parquet"
    save_canonical_parquet(canonical_df, out_parquet)
    print(f"Saved canonical Constraint Parquet to {out_parquet} ({len(canonical_df)} rows)")

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
    manifest_path = manifest_dir / "constraint.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Generated manifest: {manifest_path} ({len(manifest_df)} items)")

    return canonical_df


if __name__ == "__main__":
    df = ingest_constraint()
    print("Split counts:\n", df["split"].value_counts())
    print("Label counts:\n", df["label_raw"].value_counts())
