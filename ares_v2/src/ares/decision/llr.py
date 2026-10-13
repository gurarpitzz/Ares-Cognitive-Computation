"""Log-Likelihood Ratio (LLR) Fusion engines (Naive Bayes vs Learned Weights)."""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, f1_score

class NaiveBayesLLRFusion:
    """Unweighted Naive Bayes summation of calibrated log-odds."""
    
    def __init__(self, prior: float = 0.5):
        self.log_prior = np.log(prior / (1.0 - prior)) if 0 < prior < 1 else 0.0
        
    def fuse(self, log_odds_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        log_odds_matrix: shape (N, M) for N samples and M sources.
        Returns:
          total_log_odds: shape (N,)
          posterior_probs: shape (N,)
        """
        total_llr = self.log_prior + np.sum(log_odds_matrix, axis=1)
        probs = 1.0 / (1.0 + np.exp(-total_llr))
        return total_llr, probs

    def predict_proba(self, log_odds_matrix: np.ndarray) -> np.ndarray:
        _, p1 = self.fuse(log_odds_matrix)
        return np.column_stack([1.0 - p1, p1])

class LearnedLLRFusion:
    """Learned reliability weights via L2-regularized logistic regression on log-odds."""
    
    def __init__(self, C: float = 1.0):
        self.model = LogisticRegression(C=C, solver="lbfgs", fit_intercept=True)
        self.fitted = False
        
    def fit(self, log_odds_matrix: np.ndarray, labels: np.ndarray):
        self.model.fit(log_odds_matrix, labels)
        self.fitted = True
        return self
        
    def fuse(self, log_odds_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            # Fallback to naive bayes if not fitted
            nb = NaiveBayesLLRFusion()
            return nb.fuse(log_odds_matrix)
            
        probs = self.model.predict_proba(log_odds_matrix)[:, 1]
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        total_llr = np.log(probs / (1.0 - probs))
        return total_llr, probs

    def predict_proba(self, log_odds_matrix: np.ndarray) -> np.ndarray:
        _, p1 = self.fuse(log_odds_matrix)
        return np.column_stack([1.0 - p1, p1])
        
    @property
    def weights(self) -> np.ndarray:
        if not self.fitted:
            return np.ones(1)
        return self.model.coef_[0]
        
    @property
    def bias(self) -> float:
        if not self.fitted:
            return 0.0
        return float(self.model.intercept_[0])
