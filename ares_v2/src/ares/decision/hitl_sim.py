"""Simulated Human-in-the-Loop (HITL) Evaluation for ARES Decision Layer.

Implements static oracle and noisy analyst evaluations, escalation rate sweeps,
and 5-cycle active learning retraining against random-escalation controls.
ALL outputs from this module are explicitly marked 'SIMULATED ANALYST'.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


class SimulatedAnalyst:
    """Simulated human fact-checker / domain analyst.
    
    NOTE: All outputs are synthetic/simulated for decision-theoretic benchmarking.
    """
    
    def __init__(self, accuracy: float = 1.0, seed: int = 42, name: str = "oracle"):
        self.accuracy = float(accuracy)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.name = f"SIMULATED_{name.upper()}_a{int(accuracy * 100)}"
        
    def review(self, true_labels: np.ndarray, num_classes: int = 2) -> np.ndarray:
        """Simulate analyst reviews for escalated claims."""
        n = len(true_labels)
        if self.accuracy >= 1.0:
            return true_labels.copy()
            
        preds = true_labels.copy()
        flips = self.rng.random(n) > self.accuracy
        for i in np.where(flips)[0]:
            # Sample any class other than the true class
            alt_classes = [c for c in range(num_classes) if c != true_labels[i]]
            preds[i] = self.rng.choice(alt_classes)
            
        return preds


def sweep_static_hitl(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    escalation_rates: List[float] = [0.0, 0.05, 0.068, 0.10, 0.15, 0.20, 0.25, 0.30],
    analysts: Optional[List[SimulatedAnalyst]] = None,
    domain_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Sweep escalation rates for static simulated analyst escalation.
    
    Claims with the lowest max posterior probability are escalated first.
    """
    if analysts is None:
        analysts = [
            SimulatedAnalyst(accuracy=1.00, name="oracle"),
            SimulatedAnalyst(accuracy=0.95, name="senior_analyst"),
            SimulatedAnalyst(accuracy=0.90, name="junior_analyst"),
            SimulatedAnalyst(accuracy=0.80, name="crowd_worker"),
        ]
        
    confidences = np.max(y_prob, axis=1)
    raw_preds = np.argmax(y_prob, axis=1)
    num_classes = y_prob.shape[1]
    n = len(y_true)
    
    # Sort indices by ascending confidence (lowest confidence escalated first)
    asc_indices = np.argsort(confidences)
    
    records = []
    
    for rate in escalation_rates:
        k = int(np.round(rate * n))
        escalated_idx = set(asc_indices[:k])
        
        # Base system without escalation
        base_acc = accuracy_score(y_true, raw_preds)
        base_f1 = f1_score(y_true, raw_preds, average="macro", zero_division=0)
        
        # Calculate domain escalation rates if domains provided
        domain_rates = {}
        if domain_labels is not None:
            for d in np.unique(domain_labels):
                d_mask = (domain_labels == d)
                d_total = np.sum(d_mask)
                d_esc = sum(1 for idx in escalated_idx if domain_labels[idx] == d)
                domain_rates[f"rate_{d}"] = float(d_esc / d_total) if d_total > 0 else 0.0
                
        for analyst in analysts:
            # Analyst reviews the escalated batch
            if k > 0:
                esc_true = y_true[asc_indices[:k]]
                analyst_preds = analyst.review(esc_true, num_classes=num_classes)
                
                final_preds = raw_preds.copy()
                final_preds[asc_indices[:k]] = analyst_preds
            else:
                final_preds = raw_preds.copy()
                
            acc = accuracy_score(y_true, final_preds)
            f1 = f1_score(y_true, final_preds, average="macro", zero_division=0)
            
            # False positive reduction (class 1 as positive)
            raw_fp = np.sum((raw_preds == 1) & (y_true == 0))
            final_fp = np.sum((final_preds == 1) & (y_true == 0))
            fp_reduction = float((raw_fp - final_fp) / raw_fp) if raw_fp > 0 else 0.0
            
            rec = {
                "analyst_type": analyst.name,
                "analyst_accuracy": analyst.accuracy,
                "escalation_rate": float(rate),
                "n_escalated": k,
                "accuracy": float(acc),
                "macro_f1": float(f1),
                "delta_accuracy": float(acc - base_acc),
                "delta_f1": float(f1 - base_f1),
                "fp_reduction": float(fp_reduction),
                "provenance": "SIMULATED_ANALYST_SWEEP",
            }
            rec.update(domain_rates)
            records.append(rec)
            
    return pd.DataFrame(records)


def simulate_retraining_cycles(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_cycles: int = 5,
    escalation_batch_frac: float = 0.05,
    analyst_accuracy: float = 0.95,
    seeds: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
) -> pd.DataFrame:
    """Simulate 5 active learning retraining cycles against a random-escalation control.
    
    Verifies H5.4 and Red Flag RF4 compliance:
    Compares active escalation (lowest confidence) vs random escalation (matched sample size).
    """
    from sklearn.linear_model import LogisticRegression
    
    results = []
    
    for seed in seeds:
        rng = np.random.default_rng(seed)
        analyst = SimulatedAnalyst(accuracy=analyst_accuracy, seed=seed, name="retrain_analyst")
        
        for strategy in ["active_uncertainty", "random_control"]:
            # Initial train set: first 20% of stream
            n_total = len(X_stream)
            n_init = max(30, int(0.20 * n_total))
            
            init_idx = rng.choice(n_total, size=n_init, replace=False)
            unlabeled_mask = np.ones(n_total, dtype=bool)
            unlabeled_mask[init_idx] = False
            
            current_X_train = [X_stream[i] for i in init_idx]
            current_y_train = [y_stream[i] for i in init_idx]
            
            # Baseline model before retraining cycles
            clf = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
            clf.fit(np.array(current_X_train), np.array(current_y_train))
            
            val_probs = clf.predict_proba(X_val)
            val_preds = np.argmax(val_probs, axis=1)
            init_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            
            results.append({
                "seed": seed,
                "strategy": strategy,
                "cycle": 0,
                "n_labeled": n_init,
                "macro_f1": float(init_f1),
                "delta_f1_from_init": 0.0,
                "provenance": "SIMULATED_ACTIVE_LEARNING",
            })
            
            prev_f1 = init_f1
            
            for cycle in range(1, n_cycles + 1):
                unlabeled_idx = np.where(unlabeled_mask)[0]
                if len(unlabeled_idx) == 0:
                    break
                    
                probs = clf.predict_proba(X_stream[unlabeled_idx])
                conf = np.max(probs, axis=1)
                
                batch_size = max(5, int(escalation_batch_frac * n_total))
                batch_size = min(batch_size, len(unlabeled_idx))
                
                if strategy == "active_uncertainty":
                    # Lowest confidence
                    selected_sub = np.argsort(conf)[:batch_size]
                    selected_idx = unlabeled_idx[selected_sub]
                else:
                    # Random control
                    selected_sub = rng.choice(len(unlabeled_idx), size=batch_size, replace=False)
                    selected_idx = unlabeled_idx[selected_sub]
                    
                # Analyst labels the batch
                batch_true = y_stream[selected_idx]
                batch_labeled = analyst.review(batch_true, num_classes=len(np.unique(y_stream)))
                
                # Update pool
                unlabeled_mask[selected_idx] = False
                for idx, y_lbl in zip(selected_idx, batch_labeled):
                    current_X_train.append(X_stream[idx])
                    current_y_train.append(y_lbl)
                    
                # Retrain classifier
                clf = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
                clf.fit(np.array(current_X_train), np.array(current_y_train))
                
                # Evaluate on frozen validation set
                val_probs = clf.predict_proba(X_val)
                val_preds = np.argmax(val_probs, axis=1)
                cycle_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
                
                results.append({
                    "seed": seed,
                    "strategy": strategy,
                    "cycle": cycle,
                    "n_labeled": len(current_X_train),
                    "macro_f1": float(cycle_f1),
                    "delta_f1_from_init": float(cycle_f1 - init_f1),
                    "delta_f1_per_cycle": float(cycle_f1 - prev_f1),
                    "provenance": "SIMULATED_ACTIVE_LEARNING",
                })
                prev_f1 = cycle_f1
                
    return pd.DataFrame(results)
