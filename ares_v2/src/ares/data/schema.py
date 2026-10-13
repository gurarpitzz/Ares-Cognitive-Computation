"""Canonical data schema and validation for all ARES-Bench datasets."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Set
import pandas as pd
from pydantic import BaseModel, Field, field_validator


VALID_SPLITS: Set[str] = {"train", "val", "test", "unlabeled"}

REQUIRED_COLUMNS = [
    "item_id",
    "split",
    "label_raw",
    "label",
    "text",
    "lang",
    "date",
    "group_id",
    "media_path",
    "meta",
]


class CanonicalItem(BaseModel):
    item_id: str
    split: str
    label_raw: str
    label: Any
    text: Optional[str] = None
    lang: str = "en"
    date: Optional[str] = None
    group_id: Optional[str] = None
    media_path: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: str) -> str:
        if v not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{v}'. Must be one of {VALID_SPLITS}")
        return v


def validate_canonical_df(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the canonical ARES schema."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in canonical DataFrame: {sorted(missing)}")

    # Check for empty item_ids
    if df["item_id"].isna().any() or (df["item_id"] == "").any():
        raise ValueError("Canonical DataFrame contains null or empty item_id")

    # Check for duplicate item_ids
    if df["item_id"].duplicated().any():
        dup_count = df["item_id"].duplicated().sum()
        raise ValueError(f"Canonical DataFrame contains {dup_count} duplicate item_ids")

    # Check split values
    invalid_splits = set(df["split"].unique()) - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"Invalid splits found in DataFrame: {invalid_splits}. Allowed: {VALID_SPLITS}")


def save_canonical_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Validate and save canonical dataset to Parquet format."""
    validate_canonical_df(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure meta is JSON string if it's a dict
    df_to_save = df.copy()
    if len(df_to_save) > 0 and isinstance(df_to_save["meta"].iloc[0], (dict, list)):
        df_to_save["meta"] = df_to_save["meta"].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x))

    df_to_save.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")


def load_canonical_parquet(path: Path) -> pd.DataFrame:
    """Load and validate canonical dataset from Parquet format."""
    df = pd.read_parquet(path, engine="pyarrow")
    validate_canonical_df(df)
    return df
