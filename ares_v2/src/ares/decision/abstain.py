"""Chow optimal cost-asymmetric abstention rule (Theorem C) and risk-coverage curves."""

from typing import Dict, Any, List, Tuple
import numpy as np

class ChowAbstention:
    """Decision-theoretic Chow abstention rule with asymmetric costs."""
    
    def __init__(self, c_fp: float = 1.0, c_fn: float = 1.0, c_a: float = 0.20):
        self.c_fp = c_fp
        self.c_fn = c_fn
        self.c_a = c_a
        
        # Theorem C analytical thresholds on posterior log-odds L
        if c_a >= min(c_fp, c_fn):
            raise ValueError(f"Abstention cost c_a ({c_a}) must be strictly less than misclassification costs.")
            
        self.tau_1 = np.log((c_fp - c_a) / c_a)
        self.tau_0 = np.log(c_a / (c_fn - c_a))
        
    def decide(self, log_odds: float) -> str:
        """
        Returns:
          'PREDICT_1' if L >= tau_1
          'PREDICT_0' if L <= tau_0
          'ABSTAIN' if tau_0 < L < tau_1
        """
        if log_odds >= self.tau_1:
            return "PREDICT_1"
        elif log_odds <= self.tau_0:
            return "PREDICT_0"
        else:
            return "ABSTAIN"
            
    def evaluate_batch(
        self,
        log_odds_arr: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate decisions across a test set.
        """
        decisions = []
        for l in log_odds_arr:
            decisions.append(self.decide(l))
            
        decisions = np.array(decisions)
        n = len(labels)
        
        n_abstain = np.sum(decisions == "ABSTAIN")
        coverage = 1.0 - (n_abstain / n)
        
        covered_mask = decisions != "ABSTAIN"
        if np.sum(covered_mask) > 0:
            pred_classes = np.zeros(n, dtype=int)
            pred_classes[decisions == "PREDICT_1"] = 1
            
            fp = np.sum((pred_classes == 1) & (labels == 0) & covered_mask)
            fn = np.sum((pred_classes == 0) & (labels == 1) & covered_mask)
            tp = np.sum((pred_classes == 1) & (labels == 1) & covered_mask)
            tn = np.sum((pred_classes == 0) & (labels == 0) & covered_mask)
            
            selective_acc = (tp + tn) / np.sum(covered_mask)
        else:
            fp, fn, tp, tn = 0, 0, 0, 0
            selective_acc = 0.0
            
        total_risk = (self.c_fp * fp + self.c_fn * fn + self.c_a * n_abstain) / n
        
        return {
            "c_fp": self.c_fp,
            "c_fn": self.c_fn,
            "c_a": self.c_a,
            "tau_0": float(self.tau_0),
            "tau_1": float(self.tau_1),
            "coverage": float(coverage),
            "selective_accuracy": float(selective_acc),
            "total_risk": float(total_risk),
            "n_abstain": int(n_abstain),
            "fp": int(fp),
            "fn": int(fn)
        }

def compute_risk_coverage_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 100
) -> Dict[str, Any]:
    """
    Computes selective risk and accuracy across coverage levels from 0% to 100%.
    Uncertainty metric: |prob - 0.5|
    """
    confidences = np.abs(probs - 0.5)
    sorted_indices = np.argsort(confidences)[::-1] # most confident first
    
    coverages = []
    risks = []
    accuracies = []
    
    n = len(labels)
    for k in range(1, n + 1):
        top_k_idx = sorted_indices[:k]
        cov = k / n
        
        preds_k = (probs[top_k_idx] >= 0.5).astype(int)
        labels_k = labels[top_k_idx]
        
        acc_k = np.mean(preds_k == labels_k)
        risk_k = 1.0 - acc_k # 0-1 error
        
        coverages.append(cov)
        risks.append(risk_k)
        accuracies.append(acc_k)
        
    if hasattr(np, "trapezoid"):
        aurc = float(np.trapezoid(risks, coverages))
    else:
        aurc = float(np.trapz(risks, coverages))
    
    # Selective accuracy at fixed coverage milestones
    milestones = [0.50, 0.70, 0.80, 0.90, 1.00]
    selective_acc_milestones = {}
    for m in milestones:
        idx = int(np.clip(round(m * n) - 1, 0, n - 1))
        selective_acc_milestones[f"{int(m*100)}%"] = float(accuracies[idx])
        
    return {
        "aurc": aurc,
        "full_accuracy": float(accuracies[-1]),
        "selective_accuracy_milestones": selective_acc_milestones
    }
