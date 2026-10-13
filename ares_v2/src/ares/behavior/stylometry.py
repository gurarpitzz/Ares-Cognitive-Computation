"""Stylometric Feature Extraction and Authorship Anomaly Detection (Section VII-A).

Implements:
1. Character n-gram frequencies (n = 2, 3, 4).
2. Punctuation variance across sentences/windows: \\sigma^2 = (1/N) \\sum (p_i - \\mu_p)^2.
3. Yule's Characteristic K vocabulary richness metric:
   K = 10^4 * (\\sum_i i^2 * f_i - N) / N^2
4. One-Class Support Vector Machine (OCSVM) for authorship anomaly detection.
"""

import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def compute_yules_k(text: str) -> float:
    r"""Compute Yule's Characteristic K metric of vocabulary richness.
    
    Formula:
        K = 10^4 * (\sum_{i=1}^\infty i^2 * f_i - N) / N^2
    where f_i is the number of distinct word types appearing exactly i times,
    and N is the total token count.
    
    Args:
        text: Input text string.
    Returns:
        K: Yule's K metric (non-negative float).
    """
    tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
    n_tokens = len(tokens)
    if n_tokens <= 1:
        return 0.0

    counts = Counter(tokens)
    s2 = sum(c * c for c in counts.values())
    k = 10000.0 * (s2 - n_tokens) / (n_tokens * n_tokens)
    return float(max(0.0, k))


def compute_punctuation_variance(text: str, window_size: int = 15) -> Tuple[float, float]:
    r"""Compute punctuation frequency variance across token windows.
    
    Formula:
        \sigma^2 = \frac{1}{N} \sum_{i=1}^N (p_i - \mu_p)^2
    where p_i is the punctuation count in window i.
    
    Args:
        text: Input text string.
        window_size: Number of tokens per window.
    Returns:
        (variance, mean_rate): Punctuation variance and mean punctuation rate.
    """
    words = text.split()
    if not words:
        return 0.0, 0.0

    punct_set = set(string.punctuation)
    windows = []
    for i in range(0, len(words), window_size):
        chunk = " ".join(words[i:i + window_size])
        punct_count = sum(1 for ch in chunk if ch in punct_set)
        windows.append(punct_count)

    if len(windows) <= 1:
        # If single window, compute variance across sentence splits
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1:
            p_counts = [sum(1 for ch in s if ch in punct_set) for s in sentences]
            return float(np.var(p_counts)), float(np.mean(p_counts))
        return 0.0, float(windows[0]) if windows else 0.0

    var_p = float(np.var(windows))
    mean_p = float(np.mean(windows))
    return var_p, mean_p


class StylometryExtractor:
    """Extracts stylometric features combining n-grams, punctuation, and lexical richness."""

    def __init__(self, max_char_ngrams: int = 100):
        self.max_char_ngrams = max_char_ngrams
        self.char_vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=max_char_ngrams,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "StylometryExtractor":
        """Fit char n-gram vocabulary and standard scaler on training texts only."""
        char_counts = self.char_vectorizer.fit_transform(texts).toarray()
        
        lex_features = []
        for t in texts:
            k = compute_yules_k(t)
            p_var, p_mean = compute_punctuation_variance(t)
            n_chars = len(t)
            n_words = len(t.split())
            avg_word_len = (n_chars / max(1, n_words))
            lex_features.append([k, p_var, p_mean, avg_word_len, float(n_words)])
            
        combined = np.hstack([char_counts, np.array(lex_features)])
        self.scaler.fit(combined)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts into normalized stylometric feature vectors."""
        if not self.is_fitted:
            raise RuntimeError("StylometryExtractor must be fitted on train split first")
            
        char_counts = self.char_vectorizer.transform(texts).toarray()
        lex_features = []
        for t in texts:
            k = compute_yules_k(t)
            p_var, p_mean = compute_punctuation_variance(t)
            n_chars = len(t)
            n_words = len(t.split())
            avg_word_len = (n_chars / max(1, n_words))
            lex_features.append([k, p_var, p_mean, avg_word_len, float(n_words)])
            
        combined = np.hstack([char_counts, np.array(lex_features)])
        return self.scaler.transform(combined)


class AuthorshipAnomalyOCSVM:
    """One-Class SVM authorship-anomaly detector trained strictly on human accounts."""

    def __init__(self, nu: float = 0.05, gamma: str = "scale"):
        self.nu = nu
        self.gamma = gamma
        self.model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        self.is_fitted = False

    def fit(self, X_human: np.ndarray) -> "AuthorshipAnomalyOCSVM":
        """Train OCSVM on human training feature representations exclusively."""
        self.model.fit(X_human)
        self.is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute signed distance to separating hyperplane.
        
        Higher score = more normal (human-like).
        Negative score = anomalous (bot-like).
        Returns anomaly score where higher = more bot-like.
        """
        if not self.is_fitted:
            raise RuntimeError("OCSVM must be fitted on human accounts first")
        # decision_function returns positive for inliers (humans), negative for outliers (bots)
        # We invert so higher score = higher probability of being anomalous (bot)
        normal_scores = self.model.decision_function(X)
        anomaly_scores = -normal_scores
        return anomaly_scores

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Predict binary anomaly: 1 for bot/anomaly, 0 for normal human."""
        if not self.is_fitted:
            raise RuntimeError("OCSVM must be fitted first")
        preds = self.model.predict(X)  # 1 for inlier (human), -1 for outlier (bot)
        return (preds == -1).astype(int)
