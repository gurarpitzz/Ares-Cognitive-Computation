"""X-FACT multilingual fact-checking dataset canonical ingestion and manifest generation."""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import requests

from ares.common.paths import ARES_DATA_ROOT, ARES_ROOT
from ares.data.schema import save_canonical_parquet


BASE_URL = "https://raw.githubusercontent.com/utahnlp/x-fact/master/data/x-fact-including-en"
SPLIT_URLS = {
    "train": f"{BASE_URL}/train.all.tsv",
    "val": f"{BASE_URL}/dev.all.tsv",
    "test": f"{BASE_URL}/test.all.tsv",
    "ood": f"{BASE_URL}/ood.tsv",
    "zeroshot": f"{BASE_URL}/zeroshot.tsv",
}

BINARY_MAP = {
    "true": 1,
    "mostly true": 1,
    "mostly false": 0,
    "false": 0,
}

NATIVE_7WAY_MAP = {
    "true": 0,
    "mostly true": 1,
    "partly true/misleading": 2,
    "mostly false": 3,
    "false": 4,
    "complicated/hard to categorise": 5,
    "other": 6,
}


def download_xfact_splits(raw_dir: Path) -> None:
    """Download X-FACT TSV files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieval_meta = raw_dir / "RETRIEVAL.json"
    meta_records = {}

    for split_name, url in SPLIT_URLS.items():
        fname = Path(url).name
        target_path = raw_dir / fname
        if not target_path.exists():
            print(f"Downloading X-FACT {split_name} from {url}...")
            resp = requests.get(url, timeout=90)
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


def ingest_xfact() -> pd.DataFrame:
    """Ingest X-FACT into canonical schema and save Parquet."""
    raw_dir = ARES_DATA_ROOT / "raw" / "xfact"
    download_xfact_splits(raw_dir)

    split_files = {
        "train": raw_dir / "train.all.tsv",
        "val": raw_dir / "dev.all.tsv",
        "test": raw_dir / "test.all.tsv",
    }

    records: List[Dict] = []

    for split_name, tsv_path in split_files.items():
        if not tsv_path.exists():
            raise FileNotFoundError(f"Missing X-FACT file: {tsv_path}")

        df_raw = pd.read_csv(tsv_path, sep="\t", quoting=3, encoding_errors="replace", on_bad_lines="skip")
        print(f"Processing X-FACT {split_name}: {len(df_raw)} rows")

        for idx, row in df_raw.iterrows():
            claim = str(row.get("claim", "")).strip() if pd.notna(row.get("claim")) else ""
            if not claim or claim == "nan":
                continue

            label_raw = str(row.get("label", "")).strip().lower()
            lang = str(row.get("language", "en")).strip().lower()
            site = str(row.get("site", "")).strip()

            canonical_label = BINARY_MAP.get(label_raw, -1)
            content_hash = hashlib.sha256(claim.encode("utf-8")).hexdigest()
            item_id = f"xfact_{split_name}_{lang}_{content_hash[:16]}"

            # Gather search evidence snippets in metadata if present
            evidence_snippets = [
                str(row.get(f"evidence_{k}", "")).strip()
                for k in range(1, 6)
                if pd.notna(row.get(f"evidence_{k}")) and str(row.get(f"evidence_{k}")).strip()
            ]

            meta = {
                "language": lang,
                "fact_check_site": site,
                "native_label": label_raw,
                "native_7way_id": NATIVE_7WAY_MAP.get(label_raw, -1),
                "evidence_snippets": evidence_snippets,
                "content_sha256": content_hash,
            }

            records.append({
                "item_id": item_id,
                "split": split_name,
                "label_raw": label_raw,
                "label": canonical_label,
                "text": claim,
                "lang": lang,
                "date": None,
                "group_id": site if site else lang,
                "media_path": None,
                "meta": meta,
            })

    canonical_df = pd.DataFrame(records)
    # Deduplicate exact claims within split
    canonical_df = canonical_df.drop_duplicates(subset=["item_id"]).reset_index(drop=True)

    out_dir = ARES_DATA_ROOT / "processed" / "xfact"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "dataset.parquet"
    save_canonical_parquet(canonical_df, out_parquet)
    print(f"Saved canonical X-FACT Parquet to {out_parquet} ({len(canonical_df)} rows)")

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
    manifest_path = manifest_dir / "xfact.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Generated manifest: {manifest_path} ({len(manifest_df)} items)")

    return canonical_df


if __name__ == "__main__":
    df = ingest_xfact()
    print("Split counts:\n", df["split"].value_counts().to_dict())
    print("Languages count:", df["lang"].nunique())
    print("Top 10 languages:", df["lang"].value_counts().head(10).to_dict())
