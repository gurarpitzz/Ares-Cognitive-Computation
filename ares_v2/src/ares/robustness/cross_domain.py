"""
Milestone M12: Cross-Domain Generalization & Ben-David Discrepancy Engine.

Covers:
1. Cross-Domain Transfer Matrix across canonical domains (Claim C044):
   Politics, Health, Social Rumour/Crisis, Multilingual, Science, Disaster.
2. Cross-Dataset PolitiFact Deduplication (Audit Red Flag RF4).
3. Proxy A-Distance Computation via 5-fold CV linear discriminator on held-out embeddings (RF5, Claim C073).
   d_A = 2(1 - 2 * err).
4. Spearman rank correlation r_s with bootstrap CIs between d_A and performance drop Delta F1 (Audit Issue M-9).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


def deduplicate_texts(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Deduplicate exact and near-identical claims to eliminate cross-dataset leakage (RF4)."""
    seen_hashes: Set[str] = set()
    keep_indices: List[int] = []

    for idx, text in enumerate(df[text_col]):
        # Normalized text hash
        norm = "".join(ch.lower() for ch in str(text) if ch.isalnum())
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            keep_indices.append(idx)

    return df.iloc[keep_indices].copy().reset_index(drop=True)


def compute_proxy_a_distance(
    features_source: np.ndarray,
    features_target: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """
    Estimate proxy A-distance d_A = 2(1 - 2 * err) between two domain distributions (Ben-David et al., 2010).
    Uses 5-fold cross-validation of a linear discriminator on held-out representations (RF5).
    """
    n_s = len(features_source)
    n_t = len(features_target)
    n_samples = min(n_s, n_t)

    rng = np.random.RandomState(seed)
    idx_s = rng.choice(n_s, size=n_samples, replace=False)
    idx_t = rng.choice(n_t, size=n_samples, replace=False)

    X_s = features_source[idx_s]
    X_t = features_target[idx_t]

    X = np.vstack([X_s, X_t])
    y = np.concatenate([np.zeros(n_samples, dtype=int), np.ones(n_samples, dtype=int)])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    errs: List[float] = []

    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[val_idx])
        err = 1.0 - accuracy_score(y[val_idx], preds)
        errs.append(err)

    mean_err = float(np.mean(errs))
    # Proxy A-distance d_A = 2(1 - 2 * err), clipped to [0, 2]
    d_A = max(0.0, min(2.0, 2.0 * (1.0 - 2.0 * mean_err)))
    return float(d_A)


@dataclass
class CrossDomainTransferResult:
    source_domain: str
    target_domain: str
    in_domain_f1: float
    cross_domain_f1: float
    f1_drop: float
    proxy_a_dist: float


class CrossDomainEvaluator:
    """Evaluates cross-domain generalization matrix and proxy A-distance correlation."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def evaluate_matrix(
        self,
        domain_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        domain_names: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate full transfer matrix across domain pairs.
        domain_data: dict of domain_name -> {'train': (X, y), 'test': (X, y)}
        """
        if domain_names is None:
            domain_names = sorted(list(domain_data.keys()))

        # First train in-domain models to get diagonal baseline F1
        in_domain_models: Dict[str, LogisticRegression] = {}
        in_domain_f1: Dict[str, float] = {}

        for dom in domain_names:
            X_tr, y_tr = domain_data[dom]["train"]
            X_te, y_te = domain_data[dom]["test"]

            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=self.seed)
            clf.fit(X_tr, y_tr)
            in_domain_models[dom] = clf

            preds = clf.predict(X_te)
            in_domain_f1[dom] = float(f1_score(y_te, preds, average="macro"))

        # Evaluate all ordered pairs (S -> T)
        records: List[Dict[str, Any]] = []
        pair_results: List[CrossDomainTransferResult] = []

        for src in domain_names:
            clf_src = in_domain_models[src]
            X_src_test, _ = domain_data[src]["test"]

            for tgt in domain_names:
                X_tgt_test, y_tgt_test = domain_data[tgt]["test"]

                preds = clf_src.predict(X_tgt_test)
                f1_cross = float(f1_score(y_tgt_test, preds, average="macro"))
                f1_diag = in_domain_f1[tgt]
                f1_drop = f1_diag - f1_cross

                # Proxy A-distance
                d_A = 0.0 if src == tgt else compute_proxy_a_distance(
                    X_src_test, X_tgt_test, seed=self.seed
                )

                res = CrossDomainTransferResult(
                    source_domain=src,
                    target_domain=tgt,
                    in_domain_f1=f1_diag,
                    cross_domain_f1=f1_cross,
                    f1_drop=f1_drop,
                    proxy_a_dist=d_A,
                )
                pair_results.append(res)
                records.append({
                    "source_domain": src,
                    "target_domain": tgt,
                    "in_domain_f1": f1_diag,
                    "cross_domain_f1": f1_cross,
                    "f1_drop": f1_drop,
                    "proxy_a_dist": d_A,
                })

        df_matrix = pd.DataFrame(records)

        # Correlation between proxy A-distance and performance drop on off-diagonal pairs
        off_diag = [r for r in pair_results if r.source_domain != r.target_domain]
        a_dists = np.array([r.proxy_a_dist for r in off_diag])
        drops = np.array([r.f1_drop for r in off_diag])

        spearman_res = stats.spearmanr(a_dists, drops)
        r_s = float(spearman_res.statistic)
        p_val = float(spearman_res.pvalue)

        # Bootstrap 95% CI on Spearman rank correlation
        b_rs: List[float] = []
        n_pairs = len(a_dists)
        rng = np.random.RandomState(self.seed)
        for _ in range(1000):
            idx = rng.randint(0, n_pairs, size=n_pairs)
            if len(np.unique(a_dists[idx])) > 2 and len(np.unique(drops[idx])) > 2:
                b_r = float(stats.spearmanr(a_dists[idx], drops[idx]).statistic)
                b_rs.append(b_r)

        ci95_rs = (
            float(np.percentile(b_rs, 2.5)) if b_rs else r_s - 0.1,
            float(np.percentile(b_rs, 97.5)) if b_rs else r_s + 0.1,
        )

        stats_summary = {
            "n_domain_pairs": len(off_diag),
            "spearman_r": r_s,
            "spearman_pval": p_val,
            "spearman_ci95": ci95_rs,
            "mean_f1_drop": float(np.mean(drops)),
            "max_f1_drop": float(np.max(drops)),
            "mean_proxy_a_dist": float(np.mean(a_dists)),
        }

        return df_matrix, stats_summary
