"""Lipschitz Constant Estimation for Semantic Embeddings (Theorem B(e), M06).

Estimates the empirical Lipschitz ratio rho = ||E(c) - E(c')|| / d_sem on STS-B
and computes quantile Lipschitz constants L_q.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def estimate_lipschitz_quantiles(
    embeddings_1: np.ndarray,
    embeddings_2: np.ndarray,
    stsb_scores: np.ndarray,
    quantiles: List[float] = [0.50, 0.75, 0.90, 0.95, 0.99]
) -> Dict[str, Any]:
    """Estimate quantile Lipschitz constants L_q from STS-B pairs."""
    # Semantic distance: d_sem in [0, 1]
    d_sem = (5.0 - stsb_scores) / 5.0
    
    # Euclidean distance in embedding space
    diff = embeddings_2 - embeddings_1
    dist_emb = np.linalg.norm(diff, axis=1)
    
    # Filter non-zero semantic distance
    valid_mask = d_sem > 0.05
    valid_d_sem = d_sem[valid_mask]
    valid_dist = dist_emb[valid_mask]
    
    # Ratio rho = ||E(c) - E(c')|| / d_sem
    ratios = valid_dist / valid_d_sem
    
    # Quantiles
    l_quantiles = {}
    for q in quantiles:
        l_q = float(np.quantile(ratios, q))
        l_quantiles[f"L_{int(q*100)}"] = round(l_q, 4)
        
    # Evaluate deterministic H0 bound:
    # If benign threshold eps_sem = 0.10 (similarity >= 4.5), bound threshold = L_95 * eps_sem
    eps_benign = 0.10
    theta_det = l_quantiles["L_95"] * eps_benign
    
    # Empirical False Positive Rate on benign pairs (where d_sem <= eps_benign)
    benign_mask = d_sem <= eps_benign
    if np.sum(benign_mask) > 0:
        benign_violations = np.sum(dist_emb[benign_mask] > theta_det)
        emp_fpr = float(benign_violations / np.sum(benign_mask))
    else:
        emp_fpr = 0.0
        
    return {
        "n_pairs": int(np.sum(valid_mask)),
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "std_ratio": float(np.std(ratios)),
        "max_ratio": float(np.max(ratios)),
        "quantiles": l_quantiles,
        "deterministic_theta": round(theta_det, 4),
        "empirical_benign_fpr": round(emp_fpr, 4),
    }
