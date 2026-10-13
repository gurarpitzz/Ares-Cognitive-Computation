"""Late fusion engines for ARES-Bench.

Implements:
1. Calibrated LLR logistic regression (M05).
2. Gradient-boosted decision trees (HistGradientBoosting / GBDT).
3. Appendix-A Multilayer Perceptron (MLP) head with 1024-d projection and GELU.
4. Chow cost-asymmetric abstention rule (Theorem C) resolving Audit Issue I-11.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from ares.decision.abstain import ChowAbstention

logger = logging.getLogger(__name__)


class CalibratedLLRFusion:
    """Learned reliability weights via L2-regularized logistic regression on log-odds."""

    def __init__(self, C: float = 1.0, seed: int = 42):
        self.C = C
        self.seed = seed
        self.model = LogisticRegression(C=C, solver="lbfgs", fit_intercept=True, random_state=seed)
        self.fitted = False
        self.feature_names: List[str] = []

    def fit(self, X_llr: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "CalibratedLLRFusion":
        self.feature_names = feature_names or [f"llr_{i}" for i in range(X_llr.shape[1])]
        self.model.fit(X_llr, y)
        self.fitted = True
        return self

    def predict_proba(self, X_llr: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        probs = self.model.predict_proba(X_llr)
        return probs

    def predict(self, X_llr: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model.predict(X_llr)

    def get_log_odds(self, X_llr: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_llr)
        if probs.shape[1] == 2:
            p1 = np.clip(probs[:, 1], 1e-7, 1.0 - 1e-7)
            return np.log(p1 / (1.0 - p1))
        else:
            # Multi-class: return maximum logit
            return np.log(np.clip(probs.max(axis=1), 1e-7, 1.0 - 1e-7))

    @property
    def weights(self) -> Dict[str, float]:
        if not self.fitted:
            return {}
        coef = self.model.coef_[0]
        return {name: float(w) for name, w in zip(self.feature_names, coef)}

    @property
    def bias(self) -> float:
        if not self.fitted:
            return 0.0
        return float(self.model.intercept_[0])


class GradientBoostedFusion:
    """Non-linear ensemble fusion using HistGradientBoostingClassifier."""

    def __init__(self, max_iter: int = 100, learning_rate: float = 0.1, seed: int = 42):
        self.seed = seed
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15,
        )
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedFusion":
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model.predict(X)


class AppendixAMLPHead(nn.Module):
    """PyTorch MLP head following Appendix A of the ARES monograph.
    
    Architecture:
    Input (concatenated module features) -> Linear(1024) -> GELU -> Dropout(0.1)
    -> Linear(512) -> GELU -> Dropout(0.1) -> Linear(n_classes)
    """

    def __init__(self, in_features: int, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AppendixAMLPFusion:
    """Trainer and evaluator wrapper for AppendixAMLPHead."""

    def __init__(
        self,
        in_features: int,
        n_classes: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 32,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.in_features = in_features
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        torch.manual_seed(seed)
        self.model = AppendixAMLPHead(in_features, n_classes, dropout).to(self.device)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "AppendixAMLPFusion":
        torch.manual_seed(self.seed)
        self.model.train()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class LateFusionPipeline:
    """Master late fusion orchestrator supporting Chow abstention."""

    def __init__(
        self,
        method: str = "calibrated_llr",
        c_fp: float = 1.0,
        c_fn: float = 1.0,
        c_a: float = 0.20,
        seed: int = 42,
    ):
        self.method = method
        self.seed = seed
        self.chow = ChowAbstention(c_fp=c_fp, c_fn=c_fn, c_a=c_a)
        self.engine: Optional[Union[CalibratedLLRFusion, GradientBoostedFusion, AppendixAMLPFusion]] = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "LateFusionPipeline":
        if self.method == "calibrated_llr":
            self.engine = CalibratedLLRFusion(seed=self.seed)
            self.engine.fit(X, y, feature_names=feature_names)
        elif self.method == "gbdt":
            self.engine = GradientBoostedFusion(seed=self.seed)
            self.engine.fit(X, y)
        elif self.method == "mlp":
            n_classes = len(np.unique(y))
            self.engine = AppendixAMLPFusion(in_features=X.shape[1], n_classes=n_classes, seed=self.seed)
            self.engine.fit(X, y)
        else:
            raise ValueError(f"Unknown late fusion method: {self.method}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.engine.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.engine.predict(X)

    def decide_with_abstention(self, X: np.ndarray) -> Dict[str, Any]:
        """Applies Chow cost-asymmetric abstention rule, resolving Audit Issue I-11."""
        probs = self.predict_proba(X)
        if probs.shape[1] == 2:
            p1 = np.clip(probs[:, 1], 1e-7, 1.0 - 1e-7)
            log_odds = np.log(p1 / (1.0 - p1))
        else:
            p_max = np.clip(probs.max(axis=1), 1e-7, 1.0 - 1e-7)
            log_odds = np.log(p_max / (1.0 - p_max))

        decisions = []
        for l in log_odds:
            decisions.append(self.chow.decide(l))

        return {
            "decisions": decisions,
            "log_odds": log_odds,
            "tau_0": self.chow.tau_0,
            "tau_1": self.chow.tau_1,
            "abstention_rate": float(np.mean([d == "ABSTAIN" for d in decisions])),
        }
