"""Assumptions Testing for Theoretical Mutation Models (M06).

Tests the Gaussianity, isotropy, and independence assumptions of M01's mutation model
on empirical sentence embeddings (Hypothesis H6.1).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA


def assess_isotropy(epsilon_matrix: np.ndarray) -> Dict[str, float]:
    """Test isotropy of the empirical perturbation covariance matrix Sigma.
    
    Under perfect isotropy (Sigma = sigma^2 * I_d):
    - All eigenvalues lambda_i are equal.
    - Ratio Tr(Sigma) / (d * lambda_max) == 1.0.
    - Condition number lambda_max / lambda_min == 1.0.
    """
    n, d = epsilon_matrix.shape
    # Center the perturbations
    eps_centered = epsilon_matrix - np.mean(epsilon_matrix, axis=0, keepdims=True)
    
    # Sample covariance matrix
    cov = (eps_centered.T @ eps_centered) / (n - 1)
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    eigenvalues = np.maximum(eigenvalues, 1e-12)  # Numerical floor
    
    lambda_max = float(eigenvalues[0])
    lambda_min = float(eigenvalues[-1])
    trace_cov = float(np.sum(eigenvalues))
    
    # 1. Condition number
    cond_number = float(lambda_max / lambda_min)
    
    # 2. Ratio Tr(Sigma) / (d * lambda_max)
    # Equals 1.0 under true isotropy, << 1.0 under strong anisotropy
    anisotropy_ratio = float(trace_cov / (d * lambda_max))
    
    # 3. Roy's Effective Rank: exp(entropy(normalized eigenvalues))
    p = eigenvalues / trace_cov
    entropy = -float(np.sum(p * np.log(p + 1e-15)))
    effective_rank = float(np.exp(entropy))
    
    # 4. Top-1 and Top-10 variance explanation
    var_top1 = float(lambda_max / trace_cov)
    var_top10 = float(np.sum(eigenvalues[:10]) / trace_cov)
    
    return {
        "dimension": d,
        "n_samples": n,
        "trace": trace_cov,
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "condition_number": cond_number,
        "anisotropy_ratio": anisotropy_ratio,
        "effective_rank": effective_rank,
        "effective_rank_ratio": float(effective_rank / d),
        "variance_top1_pct": var_top1 * 100.0,
        "variance_top10_pct": var_top10 * 100.0,
        "isotropy_rejected": bool(anisotropy_ratio < 0.20 or cond_number > 10.0),
    }


def assess_gaussianity_pca(
    epsilon_matrix: np.ndarray,
    n_components: int = 10
) -> Dict[str, Any]:
    """Test Gaussianity along top principal components of epsilon."""
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(epsilon_matrix)
    
    tests = []
    p_vals = []
    for k in range(n_components):
        proj_k = projections[:, k]
        # Standardize
        proj_std = (proj_k - np.mean(proj_k)) / (np.std(proj_k) + 1e-8)
        
        # Kolmogorov-Smirnov test against N(0, 1)
        ks_stat, ks_pval = stats.kstest(proj_std, "norm")
        skew = float(stats.skew(proj_k))
        kurt = float(stats.kurtosis(proj_k))
        
        tests.append({
            "component": k + 1,
            "variance_ratio": float(pca.explained_variance_ratio_[k]),
            "ks_stat": float(ks_stat),
            "ks_pval": float(ks_pval),
            "skewness": skew,
            "kurtosis": kurt,
            "is_gaussian": bool(ks_pval > 0.01),
        })
        p_vals.append(ks_pval)
        
    return {
        "n_components_tested": n_components,
        "n_gaussian_components": sum(1 for p in p_vals if p > 0.01),
        "min_pval": float(min(p_vals)),
        "mean_pval": float(np.mean(p_vals)),
        "component_details": tests,
    }


def assess_norm_independence(
    x_matrix: np.ndarray,
    epsilon_matrix: np.ndarray
) -> Dict[str, float]:
    """Test correlation between base embedding norm ||x|| and perturbation norm ||epsilon||."""
    norm_x = np.linalg.norm(x_matrix, axis=1)
    norm_eps = np.linalg.norm(epsilon_matrix, axis=1)
    
    r_pearson, p_pearson = stats.pearsonr(norm_x, norm_eps)
    r_spearman, p_spearman = stats.spearmanr(norm_x, norm_eps)
    
    return {
        "pearson_r": float(r_pearson),
        "pearson_pval": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_pval": float(p_spearman),
        "independent": bool(abs(r_pearson) < 0.15 and p_pearson > 0.01),
    }
