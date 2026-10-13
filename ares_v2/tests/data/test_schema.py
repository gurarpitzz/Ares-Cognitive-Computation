"""Unit tests for canonical data schema."""

import pytest
import pandas as pd
from ares.data.schema import (
    REQUIRED_COLUMNS,
    validate_canonical_df,
    save_canonical_parquet,
    load_canonical_parquet,
)


def test_valid_canonical_df(tmp_path):
    data = {
        "item_id": ["item_1", "item_2"],
        "split": ["train", "test"],
        "label_raw": ["true", "false"],
        "label": [1, 0],
        "text": ["Claim one.", "Claim two."],
        "lang": ["en", "en"],
        "date": ["2026-01-01", "2026-01-02"],
        "group_id": ["g1", "g2"],
        "media_path": [None, None],
        "meta": [{}, {}],
    }
    df = pd.DataFrame(data)
    validate_canonical_df(df)

    # Test roundtrip
    p = tmp_path / "test.parquet"
    save_canonical_parquet(df, p)
    loaded = load_canonical_parquet(p)
    assert len(loaded) == 2
    assert list(loaded["item_id"]) == ["item_1", "item_2"]


def test_missing_column_raises():
    data = {
        "item_id": ["item_1"],
        "split": ["train"],
        # missing other required columns
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_canonical_df(df)


def test_invalid_split_raises():
    data = {col: ["item_1"] for col in REQUIRED_COLUMNS}
    data["split"] = ["invalid_split"]
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Invalid splits found"):
        validate_canonical_df(df)


def test_duplicate_item_id_raises():
    data = {col: ["item_1", "item_1"] for col in REQUIRED_COLUMNS}
    data["split"] = ["train", "val"]
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="duplicate item_ids"):
        validate_canonical_df(df)
