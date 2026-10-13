"""Theoretical vs Empirical ROC Comparison for Mutation Detection (M06).

Implements the exact non-central t distribution of Theorem A from M01,
computes theoretical ROC curves, and benchmarks against empirical test pairs (H6.2 & H6.3).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score


def compute_theoretical_cosine_cdf(
    tau_grid: np.ndarray,
    kappa: float,
    dimension: int = 384
) -> np.ndarray:
    """Compute exact non-central t CDF for cosine similarity Delta under noise-to-signal kappa."""
    df = dimension - 1
    delta_nct = 1.0 / (kappa + 1e-12)
    
    # Clip tau strictly within (-1, 1)
    tau_clipped = np.clip(tau_grid, -0.9999, 0.9999)
    # T_stat = tau * sqrt(df) / sqrt(1 - tau^2)
    t_stat = tau_clipped * np.sqrt(df) / np.sqrt(1.0 - tau_clipped**2)
    
    # Non-central t CDF
    cdf_values = stats.nct.cdf(t_stat, df=df, nc=delta_nct)
    
    # Clean up SciPy NCT numerical NaNs at extreme negative tails
    nan_mask = np.isnan(cdf_values)
    if np.any(nan_mask):
        cdf_values[nan_mask & (t_stat < 0)] = 0.0
        cdf_values[nan_mask & (t_stat >= 0)] = 1.0
        
    return np.clip(cdf_values, 0.0, 1.0)


def evaluate_theory_vs_empirical_roc(
    h0_cosines_train: np.ndarray,
    h1_cosines_train: np.ndarray,
    h0_cosines_test: np.ndarray,
    h1_cosines_test: np.ndarray,
    dimension: int = 384,
    n_points: int = 200
) -> Dict[str, Any]:
    """Fit theoretical parameters on train and evaluate theoretical vs empirical ROC on test."""
    # Estimate kappa_0 on H0 train: E[cos] ~ 1 - (d-1)*kappa^2 / 2
    mean_cos_0 = float(np.mean(h0_cosines_train))
    mean_cos_1 = float(np.mean(h1_cosines_train))
    
    # Invert Corollary 1: kappa ~ sqrt(2 * (1 - E[cos]) / (d - 1))
    df = dimension - 1
    kappa_0 = float(np.sqrt(max(1e-6, 2.0 * (1.0 - mean_cos_0) / df)))
    kappa_1 = float(np.sqrt(max(1e-6, 2.0 * (1.0 - mean_cos_1) / df)))
    
    # 1. Theoretical ROC Curve
    # Detection rule flags mutation if Cosine <= tau
    # False Positive Rate = P(cos <= tau | H0)
    # True Positive Rate  = P(cos <= tau | H1)
    tau_grid = np.linspace(-0.5, 0.999, n_points)
    theo_fpr = compute_theoretical_cosine_cdf(tau_grid, kappa=kappa_0, dimension=dimension)
    theo_tpr = compute_theoretical_cosine_cdf(tau_grid, kappa=kappa_1, dimension=dimension)
    
    # Sort for monotonic FPR integration
    sort_idx = np.argsort(theo_fpr)
    theo_fpr_sorted = theo_fpr[sort_idx]
    theo_tpr_sorted = theo_tpr[sort_idx]
    
    if hasattr(np, "trapezoid"):
        theo_auc = float(np.trapezoid(theo_tpr_sorted, theo_fpr_sorted))
    else:
        theo_auc = float(np.trapz(theo_tpr_sorted, theo_fpr_sorted))
        
    # 2. Empirical ROC Curve on Test Set
    # Ground truth labels: H1 (mutation) = 1, H0 (benign) = 0
    y_test = np.concatenate([np.ones(len(h1_cosines_test)), np.zeros(len(h0_cosines_test))])
    # Detector score: lower cosine => higher mutation probability => score = -cosine
    scores_cosine = np.concatenate([-h1_cosines_test, -h0_cosines_test])
    
    emp_fpr, emp_tpr, _ = roc_curve(y_test, scores_cosine)
    emp_auc = float(auc(emp_fpr, emp_tpr))
    
    delta_auc = float(abs(theo_auc - emp_auc))
    
    return {
        "dimension": dimension,
        "kappa_0_fitted": kappa_0,
        "kappa_1_fitted": kappa_1,
        "mean_cos_h0_train": mean_cos_0,
        "mean_cos_h1_train": mean_cos_1,
        "mean_cos_h0_test": float(np.mean(h0_cosines_test)),
        "mean_cos_h1_test": float(np.mean(h1_cosines_test)),
        "theoretical_auc": theo_auc,
        "empirical_auc": emp_auc,
        "delta_auc": delta_auc,
        "h6_2_confirmed": bool(delta_auc <= 0.05),
    }


def compare_cosine_vs_euclidean_power(
    emb1_h0: np.ndarray,
    emb2_h0: np.ndarray,
    emb1_h1: np.ndarray,
    emb2_h1: np.ndarray,
) -> Dict[str, float]:
    """Compare mutation detection power: Cosine vs Euclidean Distance squared (H6.3)."""
    # 1. Normalized Cosine
    norm1_h0 = emb1_h0 / np.linalg.norm(emb1_h0, axis=1, keepdims=True)
    norm2_h0 = emb2_h0 / np.linalg.norm(emb2_h0, axis=1, keepdims=True)
    cos_h0 = np.sum(norm1_h0 * norm2_h0, axis=1)
    
    norm1_h1 = emb1_h1 / np.linalg.norm(emb1_h1, axis=1, keepdims=True)
    norm2_h1 = emb2_h1 / np.linalg.norm(emb2_h1, axis=1, keepdims=True)
    cos_h1 = np.sum(norm1_h1 * norm2_h1, axis=1)
    
    # 2. Raw Euclidean distance squared ||x2 - x1||^2
    diff_h0_raw = emb2_h0 - emb1_h0
    dist2_h0_raw = np.sum(diff_h0_raw**2, axis=1)
    
    diff_h1_raw = emb2_h1 - emb1_h1
    dist2_h1_raw = np.sum(diff_h1_raw**2, axis=1)
    
    y_true = np.concatenate([np.ones(len(cos_h1)), np.zeros(len(cos_h0))])
    
    # AUC for -cosine (higher score => mutation)
    auc_cos = float(roc_auc_score(y_true, np.concatenate([-cos_h1, -cos_h0])))
    # AUC for raw Euclidean distance squared (higher distance => mutation)
    auc_dist_raw = float(roc_auc_score(y_true, np.concatenate([dist2_h1_raw, dist2_h0_raw])))
    
    return {
        "auc_cosine": auc_cos,
        "auc_euclidean_raw": auc_dist_raw,
        "delta_auc_dist_vs_cos": float(auc_dist_raw - auc_cos),
        "h6_3_confirmed": bool(auc_dist_raw >= auc_cos - 0.01),
    }
