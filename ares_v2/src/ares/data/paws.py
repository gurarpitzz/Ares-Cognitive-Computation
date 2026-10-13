"""PAWS (Paraphrase Adversaries from Word Scrambling) canonical ingestion and manifest generation."""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datasets import load_dataset

from ares.common.paths import ARES_DATA_ROOT, ARES_ROOT
from ares.data.schema import save_canonical_parquet


def ingest_paws() -> pd.DataFrame:
    """Ingest PAWS labeled_final into canonical schema and save Parquet."""
    print("Loading PAWS labeled_final via Hugging Face...")
    ds = load_dataset("paws", "labeled_final")

    records: List[Dict] = []
    split_map = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    for hf_split, canon_split in split_map.items():
        subset = ds[hf_split]
        print(f"Processing PAWS {canon_split}: {len(subset)} rows")

        for row in subset:
            s1 = str(row["sentence1"]).strip()
            s2 = str(row["sentence2"]).strip()
            lbl = int(row["label"])
            raw_id = str(row["id"])

            combined_text = f"Sentence 1: {s1} [SEP] Sentence 2: {s2}"
            content_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()
            item_id = f"paws_{canon_split}_{raw_id}"

            meta = {
                "sentence1": s1,
                "sentence2": s2,
                "original_id": raw_id,
                "content_sha256": content_hash,
            }

            records.append({
                "item_id": item_id,
                "split": canon_split,
                "label_raw": "paraphrase" if lbl == 1 else "non-paraphrase",
                "label": lbl,
                "text": combined_text,
                "lang": "en",
                "date": None,
                "group_id": None,
                "media_path": None,
                "meta": meta,
            })

    canonical_df = pd.DataFrame(records)
    out_dir = ARES_DATA_ROOT / "processed" / "paws"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "dataset.parquet"
    save_canonical_parquet(canonical_df, out_parquet)
    print(f"Saved canonical PAWS Parquet to {out_parquet} ({len(canonical_df)} rows)")

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
    manifest_path = manifest_dir / "paws.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Generated manifest: {manifest_path} ({len(manifest_df)} items)")

    return canonical_df


if __name__ == "__main__":
    df = ingest_paws()
    print("Split counts:\n", df["split"].value_counts())
    print("Label counts:\n", df["label_raw"].value_counts())
