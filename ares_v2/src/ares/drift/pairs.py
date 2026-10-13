"""Dataset Pair Extractors for Claim Drift and Mutation Analysis (M06).

Extracts and partitions H0 (benign paraphrase) and H1 (factual mutation) pairs
from canonical PAWS and STS-B datasets with disjoint splits.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ares.common.paths import ARES_DATA_ROOT

PROCESSED_DATA_DIR = ARES_DATA_ROOT / "processed"


def load_paws_pairs(sample_size: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Load and format PAWS pairs into H0 (label=1) and H1 (label=0)."""
    paws_path = PROCESSED_DATA_DIR / "paws" / "dataset.parquet"
    if not paws_path.exists():
        raise FileNotFoundError(f"Missing PAWS parquet: {paws_path}")
        
    df = pd.read_parquet(paws_path)
    
    # Parse sentences from meta JSON
    s1_list = []
    s2_list = []
    for meta_str in df["meta"]:
        try:
            m = json.loads(meta_str)
            s1_list.append(m.get("sentence1", ""))
            s2_list.append(m.get("sentence2", ""))
        except Exception as e:
            s1_list.append("")
            s2_list.append("")
            
    df["sentence1"] = s1_list
    df["sentence2"] = s2_list
    
    # Filter valid sentences
    valid_mask = (df["sentence1"].str.len() > 5) & (df["sentence2"].str.len() > 5)
    df = df[valid_mask].copy()
    
    # H0: label=1 (paraphrase, benign evolution)
    # H1: label=0 (lexical overlap, different meaning / mutation)
    df["pair_type"] = np.where(df["label"] == 1, "H0_benign", "H1_mutation")
    df["family"] = "PAWS_adversarial_lexical"
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        
    return df[["item_id", "split", "sentence1", "sentence2", "label", "pair_type", "family"]]


def load_stsb_pairs(min_score_h0: float = 4.5, max_score_h1: float = 2.0) -> pd.DataFrame:
    """Load STS-B pairs for high-confidence H0 and H1 ground truth."""
    stsb_path = PROCESSED_DATA_DIR / "stsb" / "dataset.parquet"
    if not stsb_path.exists():
        raise FileNotFoundError(f"Missing STS-B parquet: {stsb_path}")
        
    df = pd.read_parquet(stsb_path)
    
    s1_list = []
    s2_list = []
    score_list = []
    for meta_str, label_raw in zip(df["meta"], df["label_raw"]):
        try:
            m = json.loads(meta_str)
            s1_list.append(m.get("sentence1", ""))
            s2_list.append(m.get("sentence2", ""))
            score_list.append(float(label_raw))
        except Exception:
            s1_list.append("")
            s2_list.append("")
            score_list.append(float("nan"))
            
    df["sentence1"] = s1_list
    df["sentence2"] = s2_list
    df["sim_score"] = score_list
    
    # Keep only unambiguous high-similarity H0 or low-similarity H1
    h0_mask = df["sim_score"] >= min_score_h0
    h1_mask = df["sim_score"] <= max_score_h1
    
    df_filtered = df[h0_mask | h1_mask].copy()
    df_filtered["pair_type"] = np.where(df_filtered["sim_score"] >= min_score_h0, "H0_benign", "H1_mutation")
    df_filtered["family"] = "STSB_continuous"
    df_filtered["label"] = np.where(df_filtered["pair_type"] == "H0_benign", 1, 0)
    
    return df_filtered[["item_id", "split", "sentence1", "sentence2", "label", "sim_score", "pair_type", "family"]]


def get_drift_pair_sets(
    paws_sample: int = 10000,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """Retrieve combined H0/H1 pair sets partitioned into train/val/test."""
    df_paws = load_paws_pairs(sample_size=paws_sample, seed=seed)
    df_stsb = load_stsb_pairs()
    
    combined = pd.concat([df_paws, df_stsb], ignore_index=True)
    
    # Disjoint splits: train, val, test
    train_pairs = combined[combined["split"] == "train"].copy().reset_index(drop=True)
    val_pairs = combined[combined["split"] == "val"].copy().reset_index(drop=True)
    test_pairs = combined[combined["split"] == "test"].copy().reset_index(drop=True)
    
    # If val/test splits too small, perform a clean stratified split
    if len(val_pairs) < 200 or len(test_pairs) < 200:
        from sklearn.model_selection import train_test_split
        train_pairs, test_val = train_test_split(combined, test_size=0.30, random_state=seed, stratify=combined["pair_type"])
        val_pairs, test_pairs = train_test_split(test_val, test_size=0.50, random_state=seed, stratify=test_val["pair_type"])
        
    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }
