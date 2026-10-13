"""
Milestone M12: LLM-Generated Misinformation & Synthetic False-Claim Injection Engine.

Covers:
1. Performance degradation under machine-generated misinformation (Claim C037, C056).
2. Grover baseline task clarification: Provenance detection vs factual veracity verification.
3. Synthetic false-claim injection rate experiment (Claim C091):
   r_inj = |F_s| / (|T| + |F_s|) in {0.00, 0.05, 0.10, 0.20}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


@dataclass
class InjectionRunResult:
    injection_rate: float
    seed: int
    real_test_acc: float
    real_test_f1: float
    fpr: float
    fnr: float
    ece: float


def run_injection_experiment(
    X_train_real: np.ndarray,
    y_train_real: np.ndarray,
    X_synth_false: np.ndarray,
    X_test_real: np.ndarray,
    y_test_real: np.ndarray,
    injection_rates: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Train models with synthetic false claims injected at rates r = F_s / (T + F_s).
    Evaluates impact on real-world held-out test data (Claim C091).
    """
    if injection_rates is None:
        injection_rates = [0.00, 0.05, 0.10, 0.20]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    n_real = len(y_train_real)
    records: List[Dict[str, Any]] = []

    for r in injection_rates:
        rate_results = []
        for s in seeds:
            rng = np.random.RandomState(s)

            if r == 0.0:
                X_tr = X_train_real
                y_tr = y_train_real
            else:
                # Target number of synthetic false claims: F_s = (r * n_real) / (1 - r)
                n_synth = int(np.round((r * n_real) / (1.0 - r)))
                n_synth = min(n_synth, len(X_synth_false))

                synth_idx = rng.choice(len(X_synth_false), size=n_synth, replace=False)
                X_synth = X_synth_false[synth_idx]
                y_synth = np.zeros(n_synth, dtype=int)  # False label

                X_tr = np.vstack([X_train_real, X_synth])
                y_tr = np.concatenate([y_train_real, y_synth])

            # Train logistic classifier
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=s)
            clf.fit(X_tr, y_tr)

            probs = clf.predict_proba(X_test_real)[:, 1]
            preds = clf.predict(X_test_real)

            acc = float(accuracy_score(y_test_real, preds))
            f1 = float(f1_score(y_test_real, preds, average="macro"))

            # False Positive Rate: True=0 predicted as 1 (claiming false is true)
            # False Negative Rate: True=1 predicted as 0 (claiming true is false)
            fp = np.sum((y_test_real == 0) & (preds == 1))
            tn = np.sum((y_test_real == 0) & (preds == 0))
            fn = np.sum((y_test_real == 1) & (preds == 0))
            tp = np.sum((y_test_real == 1) & (preds == 1))

            fpr = float(fp / max(1, fp + tn))
            fnr = float(fn / max(1, fn + tp))
            ece = compute_ece(probs, y_test_real)

            rate_results.append({
                "injection_rate": r,
                "seed": s,
                "acc": acc,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr,
                "ece": ece,
            })

        # Aggregate across seeds
        mean_acc = float(np.mean([x["acc"] for x in rate_results]))
        std_acc = float(np.std([x["acc"] for x in rate_results], ddof=1))
        mean_f1 = float(np.mean([x["f1"] for x in rate_results]))
        std_f1 = float(np.std([x["f1"] for x in rate_results], ddof=1))
        mean_fpr = float(np.mean([x["fpr"] for x in rate_results]))
        mean_fnr = float(np.mean([x["fnr"] for x in rate_results]))
        mean_ece = float(np.mean([x["ece"] for x in rate_results]))

        records.append({
            "injection_rate": r,
            "k_seeds": len(seeds),
            "macro_f1_mean": mean_f1,
            "macro_f1_std": std_f1,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "fpr_mean": mean_fpr,
            "fnr_mean": mean_fnr,
            "ece_mean": mean_ece,
        })

    return pd.DataFrame(records)


def evaluate_machine_vs_human_misinformation(
    predict_fn: Any,
    human_texts: List[str],
    human_labels: np.ndarray,
    machine_texts: List[str],
    machine_labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate veracity degradation when encountering fluent machine-generated misinformation (C037).
    """
    human_preds = predict_fn(human_texts)
    human_f1 = float(f1_score(human_labels, human_preds, average="macro"))
    human_acc = float(accuracy_score(human_labels, human_preds))

    machine_preds = predict_fn(machine_texts)
    machine_f1 = float(f1_score(machine_labels, machine_preds, average="macro"))
    machine_acc = float(accuracy_score(machine_labels, machine_preds))

    degradation = human_f1 - machine_f1
    pct_drop = float((degradation / max(1e-6, human_f1)) * 100.0)

    return {
        "human_macro_f1": human_f1,
        "human_accuracy": human_acc,
        "machine_macro_f1": machine_f1,
        "machine_accuracy": machine_acc,
        "absolute_degradation": degradation,
        "percent_degradation": pct_drop,
    }
