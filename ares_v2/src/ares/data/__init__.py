"""ARES Data Package: Canonical schemas, dataset ingestion, and split management."""

from ares.data.schema import (
    CanonicalItem,
    REQUIRED_COLUMNS,
    VALID_SPLITS,
    load_canonical_parquet,
    save_canonical_parquet,
    validate_canonical_df,
)

__all__ = [
    "CanonicalItem",
    "REQUIRED_COLUMNS",
    "VALID_SPLITS",
    "load_canonical_parquet",
    "save_canonical_parquet",
    "validate_canonical_df",
]
