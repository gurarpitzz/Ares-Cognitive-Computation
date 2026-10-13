"""Temporal Posting Pattern Analysis and Hidden Markov Model Modeling (Section VII-B).

Implements:
1. Inter-post intervals: \\Delta t_i = t_{i+1} - t_i.
2. Interval entropy: H(\\Delta t) = -\\sum_b p_b \\log_2 p_b.
3. Burstiness metric: B = (\\sigma - \\mu) / (\\sigma + \\mu) (Goh & Barabasi, 2008).
4. Coefficient of variation: CV = \\sigma / \\mu.
5. Autocorrelation periodicity and peak detection.
6. Hidden Markov Models (HMM with Gaussian emissions on log-intervals) for bot vs human.
7. Length-normalized log-likelihood ratio:
   \\Lambda = (\\log P(O | \\lambda_bot) - \\log P(O | \\lambda_human)) / |O|
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from hmmlearn import hmm


def compute_inter_post_intervals(timestamps: np.ndarray) -> np.ndarray:
    """Compute sorted inter-post intervals in seconds.
    
    Args:
        timestamps: Array of unix timestamps or sequential seconds.
    Returns:
        intervals: Array of non-negative differences between consecutive timestamps.
    """
    if len(timestamps) <= 1:
        return np.array([], dtype=float)
    sorted_ts = np.sort(np.asarray(timestamps, dtype=float))
    diffs = np.diff(sorted_ts)
    return np.maximum(0.0, diffs)


def compute_interval_entropy(intervals: np.ndarray, num_bins: int = 20) -> float:
    r"""Compute Shannon entropy of binned posting intervals.
    
    Formula:
        H(\Delta t) = -\sum_b p_b \log_2 p_b
    
    Args:
        intervals: Array of inter-post intervals in seconds.
        num_bins: Number of histogram bins.
    Returns:
        entropy: Non-negative Shannon entropy in bits.
    """
    if len(intervals) <= 1:
        return 0.0
    
    # Use log-spaced intervals to capture orders of magnitude (seconds to days)
    log_intervals = np.log1p(intervals)
    min_val, max_val = float(np.min(log_intervals)), float(np.max(log_intervals))
    if max_val - min_val < 1e-6:
        return 0.0

    counts, _ = np.histogram(log_intervals, bins=num_bins, range=(min_val, max_val))
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_burstiness_and_cv(intervals: np.ndarray) -> Tuple[float, float]:
    r"""Compute coefficient of variation (CV) and Goh-Barabasi burstiness metric (B).
    
    Formulas:
        CV = \sigma / \mu
        B = (\sigma - \mu) / (\sigma + \mu) = (CV - 1) / (CV + 1)
        B \in [-1, 1]:
            B = -1: completely regular / periodic (automated)
            B = 0: memoryless Poisson process
            B > 0: bursty human dynamics
    
    Args:
        intervals: Array of inter-post intervals.
    Returns:
        (burstiness, cv): Burstiness B and coefficient of variation CV.
    """
    if len(intervals) <= 1:
        return 0.0, 0.0
    mu = float(np.mean(intervals))
    sigma = float(np.std(intervals))
    if mu <= 1e-9:
        return -1.0, 0.0
    cv = sigma / mu
    burstiness = (sigma - mu) / (sigma + mu) if (sigma + mu) > 1e-9 else 0.0
    return float(burstiness), float(cv)


def compute_autocorrelation_peak(intervals: np.ndarray, max_lag: int = 10) -> Tuple[float, int]:
    """Compute the maximum non-zero lag autocorrelation of posting intervals.
    
    Args:
        intervals: Array of inter-post intervals.
        max_lag: Maximum lag to examine.
    Returns:
        (peak_corr, peak_lag): Peak autocorrelation value and corresponding lag.
    """
    n = len(intervals)
    if n <= 3:
        return 0.0, 1
    
    vals = intervals - np.mean(intervals)
    variance = np.var(intervals)
    if variance <= 1e-9:
        # Perfectly constant intervals -> perfect autocorrelation
        return 1.0, 1

    effective_max_lag = min(max_lag, n // 2)
    if effective_max_lag < 1:
        return 0.0, 1

    autocorrs = []
    for lag in range(1, effective_max_lag + 1):
        corr = np.mean(vals[:-lag] * vals[lag:]) / variance
        autocorrs.append((float(corr), lag))

    if not autocorrs:
        return 0.0, 1
    
    best_corr, best_lag = max(autocorrs, key=lambda x: abs(x[0]))
    return best_corr, best_lag


class TemporalHMMAuditor:
    r"""Gaussian Hidden Markov Model for temporal posting behavior (Section VII-B).
    
    Fits two independent HMMs with Gaussian emissions on log-transformed
    inter-post intervals \log(\Delta t + 1):
      1. \lambda_human: Trained on verified human timelines.
      2. \lambda_bot: Trained on automated bot timelines.
    
    Calculates the normalized Log-Likelihood Ratio (LLR):
        \Lambda(O) = (\log P(O | \lambda_bot) - \log P(O | \lambda_human)) / |O|
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 50,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.human_hmm = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self.bot_hmm = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state + 1,
        )
        self.is_fitted = False

    def _prepare_sequences(
        self, account_intervals: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[int]]:
        """Flatten sequences and track sequence lengths for hmmlearn."""
        concatenated = []
        lengths = []
        for intervals in account_intervals:
            if len(intervals) == 0:
                continue
            log_ints = np.log1p(np.asarray(intervals, dtype=float)).reshape(-1, 1)
            concatenated.append(log_ints)
            lengths.append(len(log_ints))

        if not concatenated:
            return np.zeros((0, 1)), []
        return np.vstack(concatenated), lengths

    def fit(
        self,
        human_account_intervals: List[np.ndarray],
        bot_account_intervals: List[np.ndarray],
    ) -> "TemporalHMMAuditor":
        """Fit human and bot HMMs on training interval sequences."""
        X_h, len_h = self._prepare_sequences(human_account_intervals)
        X_b, len_b = self._prepare_sequences(bot_account_intervals)

        if len(len_h) == 0 or len(len_b) == 0:
            raise ValueError("Both human and bot training sets must contain valid interval sequences")

        self.human_hmm.fit(X_h, len_h)
        self.bot_hmm.fit(X_b, len_b)
        self.is_fitted = True
        return self

    def compute_log_likelihood_ratio(self, intervals: np.ndarray) -> float:
        r"""Compute normalized log-likelihood ratio \Lambda for an account's intervals.
        
        Args:
            intervals: Array of inter-post intervals for a single account.
        Returns:
            llr: Normalized log-likelihood ratio (\Lambda). Positive = bot-like, Negative = human-like.
        """
        if not self.is_fitted:
            raise RuntimeError("TemporalHMMAuditor must be fitted before computing LLR")

        if len(intervals) == 0:
            return 0.0

        obs = np.log1p(np.asarray(intervals, dtype=float)).reshape(-1, 1)
        n = len(obs)

        try:
            log_p_human = self.human_hmm.score(obs)
        except Exception:
            log_p_human = -1e4 * n

        try:
            log_p_bot = self.bot_hmm.score(obs)
        except Exception:
            log_p_bot = -1e4 * n

        llr = (log_p_bot - log_p_human) / max(1, n)
        return float(np.clip(llr, -50.0, 50.0))

    def extract_temporal_features(self, intervals: np.ndarray) -> Dict[str, float]:
        """Extract complete 5-dimensional temporal feature dictionary for an account."""
        if len(intervals) == 0:
            return {
                "interval_mean": 0.0,
                "interval_entropy": 0.0,
                "burstiness": 0.0,
                "cv": 0.0,
                "autocorr_peak": 0.0,
                "hmm_llr": 0.0,
            }

        burstiness, cv = compute_burstiness_and_cv(intervals)
        entropy = compute_interval_entropy(intervals)
        peak_corr, _ = compute_autocorrelation_peak(intervals)
        llr = self.compute_log_likelihood_ratio(intervals) if self.is_fitted else 0.0

        return {
            "interval_mean": float(np.mean(intervals)),
            "interval_entropy": entropy,
            "burstiness": burstiness,
            "cv": cv,
            "autocorr_peak": peak_corr,
            "hmm_llr": llr,
        }
