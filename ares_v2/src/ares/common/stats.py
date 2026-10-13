"""Statistical evaluation and hypothesis testing library for ARES."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats
from sklearn.metrics import f1_score, precision_recall_fscore_support


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard macro-averaged F1 score via scikit-learn."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def macro_f1_independent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Independent hand-written macro-averaged F1 score for auditor cross-checks.
    
    Computes per-class F1 = 2*TP / (2*TP + FP + FN) and averages over unique classes in y_true.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    if len(classes) == 0:
        return 0.0
    
    f1_list = []
    for c in classes:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        
        denom = 2 * tp + fp + fn
        if denom == 0:
            f1 = 0.0
        else:
            f1 = (2.0 * tp) / denom
        f1_list.append(f1)
        
    return float(np.mean(f1_list))


def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Per-class Precision, Recall, F1 and Support."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    out = {}
    for idx, c in enumerate(classes):
        out[str(c)] = {
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f[idx]),
            "support": int(s[idx])
        }
    return out


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    B: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float]:
    """Compute non-parametric percentile bootstrap confidence interval [ci_lower, ci_upper]."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = np.arange(n)
    
    boot_stats = np.empty(B, dtype=np.float64)
    for b in range(B):
        sample_idx = rng.choice(indices, size=n, replace=True)
        boot_stats[b] = metric_fn(y_true[sample_idx], y_pred[sample_idx])
        
    lower = float(np.percentile(boot_stats, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def paired_bootstrap_delta(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    B: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """Paired bootstrap comparison between Model A and Model B.
    
    Returns:
        (delta, ci_lower, ci_upper, p_value) where delta = Metric(B) - Metric(A)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = np.arange(n)
    
    obs_a = metric_fn(y_true, pred_a)
    obs_b = metric_fn(y_true, pred_b)
    obs_delta = obs_b - obs_a
    
    boot_deltas = np.empty(B, dtype=np.float64)
    for b in range(B):
        sample_idx = rng.choice(indices, size=n, replace=True)
        m_a = metric_fn(y_true[sample_idx], pred_a[sample_idx])
        m_b = metric_fn(y_true[sample_idx], pred_b[sample_idx])
        boot_deltas[b] = m_b - m_a
        
    ci_lower = float(np.percentile(boot_deltas, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_deltas, 100.0 * (1.0 - alpha / 2.0)))
    
    # Two-sided empirical p-value for H0: delta == 0
    p_pos = np.mean(boot_deltas <= 0)
    p_neg = np.mean(boot_deltas >= 0)
    p_val = float(min(1.0, 2.0 * min(p_pos, p_neg)))
    
    return float(obs_delta), ci_lower, ci_upper, p_val


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> Tuple[int, float]:
    """Exact McNemar's test for paired binary outcomes.
    
    Table:
                B Correct   B Incorrect
    A Correct       n00         n01 (b)
    A Incorrect     n10 (c)     n11
    
    Returns:
        (discordant_count, two_sided_p_value)
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_discordant = b + c
    if n_discordant == 0:
        return 0, 1.0
    
    # Under H0, b ~ Binomial(n_discordant, 0.5)
    p_val = float(stats.binomtest(b, n_discordant, p=0.5, alternative="two-sided").pvalue)
    return n_discordant, p_val


def paired_t(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Paired Student's t-test across K seeds."""
    res = stats.ttest_rel(a, b)
    return float(res.statistic), float(res.pvalue)


def wilcoxon_exact(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank test using exact distribution."""
    res = stats.wilcoxon(a, b, alternative="two-sided", method="exact")
    return float(res.statistic), float(res.pvalue)


def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d_z effect size for paired samples: mean(diff) / std(diff, ddof=1)."""
    diff = np.asarray(b) - np.asarray(a)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Holm-Bonferroni step-down procedure for family-wise error rate control.
    
    Returns:
        (adjusted_p_values, reject_hypotheses)
    """
    m = len(p_values)
    if m == 0:
        return [], []
    
    sorted_indices = np.argsort(p_values)
    adj_p = [0.0] * m
    reject = [False] * m
    
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_values[idx]
        multiplier = m - rank
        raw_adj = p * multiplier
        running_max = max(running_max, raw_adj)
        adj_p[idx] = min(1.0, running_max)
        
    for idx, p_adj in enumerate(adj_p):
        reject[idx] = (p_adj < alpha)
        
    return adj_p, reject


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) with equal-width confidence bins."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confs = np.max(probs, axis=1) if probs.ndim == 2 else probs
    preds = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs >= 0.5).astype(int)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(labels)
    
    for i in range(n_bins):
        in_bin = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin] == preds[in_bin])
            avg_confidence_in_bin = np.mean(confs[in_bin])
            ece_val += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece_val)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score (mean squared error of predicted probabilities)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.ndim == 1:
        return float(np.mean((probs - labels) ** 2))
    elif probs.ndim == 2:
        # Multi-class one-hot brier
        n_classes = probs.shape[1]
        one_hot = np.eye(n_classes)[labels]
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return 0.0


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability diagram coordinates: bin accuracies, confidences, and counts."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confs = np.max(probs, axis=1) if probs.ndim == 2 else probs
    preds = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs >= 0.5).astype(int)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        in_bin = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        bin_counts[i] = int(np.sum(in_bin))
        if bin_counts[i] > 0:
            bin_accs[i] = np.mean(labels[in_bin] == preds[in_bin])
            bin_confs[i] = np.mean(confs[in_bin])
            
    return bin_accs, bin_confs, bin_counts


def interaction_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    preds_full: np.ndarray,
    preds_minusA: np.ndarray,
    preds_minusB: np.ndarray,
    preds_minusAB: np.ndarray,
    y_true: np.ndarray,
    B: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Compute interaction effect Delta_AB - Delta_A - Delta_B with percentile bootstrap CI.
    
    Interaction = (F(full) - F(minusAB)) - (F(full) - F(minusA)) - (F(full) - F(minusB))
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = np.arange(n)
    
    def calc_interaction(idx):
        f = metric_fn(y_true[idx], preds_full[idx])
        ma = metric_fn(y_true[idx], preds_minusA[idx])
        mb = metric_fn(y_true[idx], preds_minusB[idx])
        mab = metric_fn(y_true[idx], preds_minusAB[idx])
        delta_ab = f - mab
        delta_a = f - ma
        delta_b = f - mb
        return delta_ab - delta_a - delta_b
        
    obs_val = calc_interaction(indices)
    boot_vals = np.empty(B, dtype=np.float64)
    for b in range(B):
        sample_idx = rng.choice(indices, size=n, replace=True)
        boot_vals[b] = calc_interaction(sample_idx)
        
    ci_lower = float(np.percentile(boot_vals, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_vals, 100.0 * (1.0 - alpha / 2.0)))
    return float(obs_val), ci_lower, ci_upper
