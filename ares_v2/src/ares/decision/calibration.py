"""Probability calibration operators (Platt scaling, Temperature scaling, ECE)."""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier Score."""
    return float(brier_score_loss(labels, probs))


class PlattScaler:
    """Logistic regression (Platt) calibration."""
    
    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver="lbfgs")
        self.fitted = False
        
    def fit(self, logits: np.ndarray, labels: np.ndarray):
        logits_2d = logits.reshape(-1, 1)
        self.model.fit(logits_2d, labels)
        self.fitted = True
        return self
        
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if not self.fitted:
            # Fallback to sigmoid
            return 1.0 / (1.0 + np.exp(-logits))
        logits_2d = logits.reshape(-1, 1)
        return self.model.predict_proba(logits_2d)[:, 1]

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return self.predict_proba(logits)
        
    def to_log_odds(self, logits: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        probs = self.predict_proba(logits)
        probs = np.clip(probs, eps, 1.0 - eps)
        return np.log(probs / (1.0 - probs))

class TemperatureScaler:
    """Single-parameter temperature scaling on logits."""
    
    def __init__(self):
        self.temperature = 1.0
        
    def fit(self, logits: np.ndarray, labels: np.ndarray):
        def nll_obj(T):
            temp = max(T[0], 0.01)
            scaled_logits = logits / temp
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            # Binary cross entropy
            loss = -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
            return loss
            
        res = minimize(nll_obj, x0=[1.0], bounds=[(0.05, 10.0)])
        self.temperature = float(res.x[0])
        return self
        
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))
        
    def to_log_odds(self, logits: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        probs = self.predict_proba(logits)
        probs = np.clip(probs, eps, 1.0 - eps)
        return np.log(probs / (1.0 - probs))
