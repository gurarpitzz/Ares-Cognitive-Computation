"""Direct HTTP streaming model downloader for offline Hugging Face models."""

import os
import sys
import time
import urllib.request
from pathlib import Path

from ares.common.paths import ARES_CACHE_ROOT

MODEL_FILES = {
    "roberta-base": [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    ],
    "microsoft/deberta-v3-base": [
        "config.json",
        "tokenizer_config.json",
        "spm.model",
        "pytorch_model.bin",
    ],
    "microsoft/mdeberta-v3-base": [
        "config.json",
        "tokenizer_config.json",
        "spm.model",
        "pytorch_model.bin",
    ],
}


def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  [Exists] {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    print(f"  [Downloading] {url} -> {dest_path.name}...", flush=True)
    t0 = time.time()

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ARES/2.0"}
    )

    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f:
        total_len = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1 MB blocks
        last_log = time.time()

        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_log >= 2.0 or (total_len > 0 and downloaded == total_len):
                mb = downloaded / (1024 * 1024)
                total_mb = total_len / (1024 * 1024)
                pct = (downloaded / total_len * 100) if total_len > 0 else 0
                speed = mb / max(now - t0, 0.01)
                print(f"    Progress: {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", flush=True)
                last_log = now

    temp_path.replace(dest_path)
    dur = time.time() - t0
    final_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"  [Done] {dest_path.name} ({final_mb:.2f} MB in {dur:.1f}s)", flush=True)


def download_model(model_name: str):
    clean_name = model_name.replace("/", "_")
    target_dir = ARES_CACHE_ROOT / "models" / clean_name
    target_dir.mkdir(parents=True, exist_ok=True)

    files = MODEL_FILES.get(model_name, ["config.json", "tokenizer_config.json", "model.safetensors"])
    print(f"\n=======================================================", flush=True)
    print(f"Downloading {model_name} to {target_dir}", flush=True)
    print(f"=======================================================", flush=True)

    base_url = f"https://huggingface.co/{model_name}/resolve/main"
    for fname in files:
        url = f"{base_url}/{fname}"
        dest = target_dir / fname
        download_file(url, dest)

    print(f"\nFinished downloading {model_name} -> {target_dir}\n", flush=True)
    return target_dir


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    models = sys.argv[1:] if len(sys.argv) > 1 else ["roberta-base"]
    for m in models:
        download_model(m)


if __name__ == "__main__":
    main()
