"""Unit tests for manifest and sha256 utilities."""

import pandas as pd
from ares.common.manifest import build_manifest, manifest_hash, sha256_file


def test_sha256_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello ares", encoding="utf-8")
    digest = sha256_file(f)
    assert len(digest) == 64
    assert isinstance(digest, str)


def test_build_manifest():
    df = pd.DataFrame({
        "id": ["c", "a", "b"],
        "split": ["val", "train", "test"],
        "label": [0, 1, 0]
    })
    m = build_manifest(df, id_col="id", split_col="split", label_col="label")
    assert list(m.columns) == ["item_id", "split", "label"]
    h1 = manifest_hash(m)
    h2 = manifest_hash(m)
    assert h1 == h2
    assert h1.startswith("sha256:")
