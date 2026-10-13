"""Pre-fetch transformer models into local Hugging Face cache on Drive D."""

import os
import sys
import time
from pathlib import Path

from ares.common.paths import ARES_CACHE_ROOT

# Ensure cache is on D: drive
cache_dir = ARES_CACHE_ROOT / "huggingface"
os.environ["HF_HOME"] = str(cache_dir)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download


def prefetch(model_id: str):
    print(f"\n[Prefetch] Downloading {model_id} into {cache_dir}...", flush=True)
    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir / "hub"),
            resume_download=True,
        )
        dur = time.time() - t0
        print(f"[Prefetch] Successfully fetched {model_id} in {dur:.1f}s -> {path}", flush=True)
    except Exception as e:
        print(f"[Prefetch] Error fetching {model_id}: {e}", flush=True)
        raise


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    models = [
        "roberta-base",
        "microsoft/deberta-v3-base",
        "microsoft/mdeberta-v3-base",
    ]
    if len(sys.argv) > 1:
        models = sys.argv[1:]

    for m in models:
        prefetch(m)
    print("\nAll requested models pre-fetched successfully.", flush=True)


if __name__ == "__main__":
    main()
