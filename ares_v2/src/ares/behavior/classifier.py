"""Multimodal Behavioral Classifier and Feature Fusion (Section VII).

Combines:
1. Stylometric Authorship Anomaly (Section VII-A)
2. Temporal Posting Dynamics & HMM LLR (Section VII-B)
3. Profile Metadata Signals (Section VII-B/C)
4. Social Graph Topology & Louvain Communities (Section VII-C)

Supports rigorous feature family ablation to validate Hypothesis H9.1 and H9.3.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class MultimodalBehaviorClassifier:
    r"""Fused Behavioral Classifier uniting stylometry, temporal, profile, and graph signals.
    
    Provides:
    - Probabilistic bot prediction P(bot | x).
    - Family-level ablation masks: 'all', 'stylometry_only', 'temporal_only',
      'profile_only', 'graph_only', 'no_graph', 'no_temporal'.
    - Calibrated posterior probabilities for Section VIII Bayesian fusion.
    """

    SUPPORTED_FAMILIES = ["stylometry", "temporal", "profile", "graph"]

    def __init__(
        self,
        classifier_type: str = "gradient_boosting",
        active_families: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.classifier_type = classifier_type
        self.active_families = active_families or self.SUPPORTED_FAMILIES
        self.random_state = random_state

        if classifier_type == "gradient_boosting":
            self.clf = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=31,
                random_state=random_state,
            )
        elif classifier_type == "logistic_regression":
            self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        elif classifier_type == "random_forest":
            self.clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_slices: Dict[str, Tuple[int, int]] = {}

    def _assemble_features(
        self,
        stylometry_feats: Optional[np.ndarray] = None,
        temporal_feats: Optional[np.ndarray] = None,
        profile_feats: Optional[np.ndarray] = None,
        graph_feats: Optional[np.ndarray] = None,
        is_fitting: bool = False,
    ) -> np.ndarray:
        """Concatenate selected active feature families into unified matrix."""
        selected_matrices = []
        curr_dim = 0

        family_data = {
            "stylometry": stylometry_feats,
            "temporal": temporal_feats,
            "profile": profile_feats,
            "graph": graph_feats,
        }

        for fam in self.SUPPORTED_FAMILIES:
            if fam in self.active_families and family_data[fam] is not None:
                mat = family_data[fam]
                if mat.ndim == 1:
                    mat = mat.reshape(-1, 1)
                selected_matrices.append(mat)
                dim = mat.shape[1]
                if is_fitting:
                    self.feature_slices[fam] = (curr_dim, curr_dim + dim)
                curr_dim += dim

        if not selected_matrices:
            raise ValueError("No active feature matrices provided to classifier")

        return np.hstack(selected_matrices)

    def fit(
        self,
        y_train: np.ndarray,
        stylometry_feats: Optional[np.ndarray] = None,
        temporal_feats: Optional[np.ndarray] = None,
        profile_feats: Optional[np.ndarray] = None,
        graph_feats: Optional[np.ndarray] = None,
    ) -> "MultimodalBehaviorClassifier":
        """Fit scaler and base classifier on training features."""
        X_train = self._assemble_features(
            stylometry_feats=stylometry_feats,
            temporal_feats=temporal_feats,
            profile_feats=profile_feats,
            graph_feats=graph_feats,
            is_fitting=True,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.clf.fit(X_train_scaled, y_train)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        stylometry_feats: Optional[np.ndarray] = None,
        temporal_feats: Optional[np.ndarray] = None,
        profile_feats: Optional[np.ndarray] = None,
        graph_feats: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict class probabilities: returns (N, 2) array [P(human), P(bot)]."""
        if not self.is_fitted:
            raise RuntimeError("MultimodalBehaviorClassifier must be fitted before predict_proba")

        X = self._assemble_features(
            stylometry_feats=stylometry_feats,
            temporal_feats=temporal_feats,
            profile_feats=profile_feats,
            graph_feats=graph_feats,
            is_fitting=False,
        )

        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)

    def predict(
        self,
        stylometry_feats: Optional[np.ndarray] = None,
        temporal_feats: Optional[np.ndarray] = None,
        profile_feats: Optional[np.ndarray] = None,
        graph_feats: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary bot labels (1 = bot, 0 = human)."""
        probs = self.predict_proba(
            stylometry_feats=stylometry_feats,
            temporal_feats=temporal_feats,
            profile_feats=profile_feats,
            graph_feats=graph_feats,
        )[:, 1]
        return (probs >= threshold).astype(int)
