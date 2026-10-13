"""CoAID (COVID-19 Healthcare Misinformation) canonical ingestion and grouped split generation."""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import requests

from ares.common.paths import ARES_DATA_ROOT, ARES_ROOT
from ares.data.schema import save_canonical_parquet


RELEASES = ["05-01-2020", "07-01-2020", "09-01-2020", "11-01-2020"]
FILES = [
    ("ClaimFakeCOVID-19.csv", "claim", "fake"),
    ("ClaimRealCOVID-19.csv", "claim", "real"),
    ("NewsFakeCOVID-19.csv", "news", "fake"),
    ("NewsRealCOVID-19.csv", "news", "real"),
]
BASE_URL = "https://raw.githubusercontent.com/cuilimeng/CoAID/master"


def extract_domain(url_str: str) -> str:
    """Extract domain from URL for grouped split assignment."""
    if not url_str or pd.isna(url_str):
        return "unknown"
    try:
        parsed = urlparse(str(url_str))
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc if netloc else "unknown"
    except Exception:
        return "unknown"


def download_coaid_files(raw_dir: Path) -> None:
    """Download CoAID CSV files across available releases."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieval_meta = raw_dir / "RETRIEVAL.json"
    meta_records = {}

    for rel in RELEASES:
        rel_dir = raw_dir / rel
        rel_dir.mkdir(parents=True, exist_ok=True)
        for fname, _, _ in FILES:
            target_path = rel_dir / fname
            if not target_path.exists():
                url = f"{BASE_URL}/{rel}/{fname}"
                print(f"Downloading CoAID {rel}/{fname}...")
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    target_path.write_bytes(resp.content)
                    sha256 = hashlib.sha256(resp.content).hexdigest()
                    meta_records[f"{rel}_{fname}"] = {
                        "source_url": url,
                        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "sha256": sha256,
                        "bytes": len(resp.content),
                    }
                else:
                    print(f"Notice: {rel}/{fname} returned status {resp.status_code}")

    if meta_records and not retrieval_meta.exists():
        retrieval_meta.write_text(json.dumps(meta_records, indent=2))


def ingest_coaid() -> pd.DataFrame:
    """Ingest CoAID into canonical schema and grouped 70/10/20 train/val/test split."""
    raw_dir = ARES_DATA_ROOT / "raw" / "coaid"
    download_coaid_files(raw_dir)

    all_raw_rows = []

    for rel in RELEASES:
        rel_dir = raw_dir / rel
        for fname, content_type, label_str in FILES:
            fpath = rel_dir / fname
            if not fpath.exists():
                continue
            try:
                df_part = pd.read_csv(fpath)
            except Exception as e:
                print(f"Warning reading {fpath}: {e}")
                continue

            for _, row in df_part.iterrows():
                title = str(row.get("title", "")).strip() if pd.notna(row.get("title")) else ""
                content = str(row.get("content", "")).strip() if pd.notna(row.get("content")) else ""
                url = str(row.get("news_url", row.get("fact_check_url", "")))

                text_body = title
                if not text_body and content:
                    text_body = content[:500]

                if not text_body or text_body == "nan":
                    continue

                domain = extract_domain(url)
                all_raw_rows.append({
                    "release": rel,
                    "file": fname,
                    "type": content_type,
                    "label_raw": label_str,
                    "label": 1 if label_str == "real" else 0,
                    "text": text_body,
                    "url": url,
                    "group_id": domain if domain != "unknown" else f"ungrouped_{hashlib.md5(text_body.encode()).hexdigest()[:8]}",
                    "content_sha256": hashlib.sha256(text_body.encode("utf-8")).hexdigest(),
                })

    df_combined = pd.DataFrame(all_raw_rows)
    # Deduplicate exact text bodies within CoAID
    df_unique = df_combined.drop_duplicates(subset=["content_sha256"]).copy().reset_index(drop=True)
    print(f"Total unique CoAID items collected: {len(df_unique)} (from {len(df_combined)} raw entries)")

    # Grouped stratified split 70/10/20 by group_id
    rng = np.random.RandomState(20261013)
    groups = df_unique["group_id"].unique()
    rng.shuffle(groups)

    n_groups = len(groups)
    n_train = int(0.70 * n_groups)
    n_val = int(0.10 * n_groups)

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    def assign_split(g):
        if g in train_groups:
            return "train"
        elif g in val_groups:
            return "val"
        else:
            return "test"

    df_unique["split"] = df_unique["group_id"].apply(assign_split)

    # Build canonical records
    records = []
    for idx, row in df_unique.iterrows():
        item_id = f"coaid_{row['split']}_{row['content_sha256'][:16]}"
        meta = {
            "release": row["release"],
            "type": row["type"],
            "source_url": row["url"],
            "domain": row["group_id"],
            "content_sha256": row["content_sha256"],
        }
        records.append({
            "item_id": item_id,
            "split": row["split"],
            "label_raw": row["label_raw"],
            "label": row["label"],
            "text": row["text"],
            "lang": "en",
            "date": None,
            "group_id": row["group_id"],
            "media_path": None,
            "meta": meta,
        })

    canonical_df = pd.DataFrame(records)
    out_dir = ARES_DATA_ROOT / "processed" / "coaid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "dataset.parquet"
    save_canonical_parquet(canonical_df, out_parquet)
    print(f"Saved canonical CoAID Parquet to {out_parquet} ({len(canonical_df)} rows)")

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
    manifest_path = manifest_dir / "coaid.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Generated manifest: {manifest_path} ({len(manifest_df)} items)")

    return canonical_df


if __name__ == "__main__":
    df = ingest_coaid()
    print("Split counts:\n", df["split"].value_counts())
    print("Label counts:\n", df["label_raw"].value_counts())
