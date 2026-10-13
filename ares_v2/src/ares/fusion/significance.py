"""Statistical significance testing battery for ARES-Bench (Claims C051, C052).

Executes:
1. Instance-level tests: Paired percentile bootstrap delta (B=10000) and exact McNemar test.
2. Seed-level tests across K=10 seeds: Paired t-test and exact Wilcoxon signed-rank test.
3. Transparent documentation of why K=5 Wilcoxon p < 0.01 was mathematically impossible (Audit Issue M-1).
4. Standardized effect size Cohen's d_z with analytic confidence intervals.
5. Family-wise Holm-Bonferroni step-down multiple testing correction.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
from sklearn.metrics import f1_score

from ares.common.stats import (
    bootstrap_ci,
    cohens_dz,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_delta,
    paired_t,
    wilcoxon_exact,
)

logger = logging.getLogger(__name__)


def prove_wilcoxon_k5_impossibility() -> Dict[str, Any]:
    """Provides analytical proof that K=5 Wilcoxon p < 0.01 is mathematically impossible."""
    # Under H0, there are 2^K equally likely signed rank combinations
    # The most extreme outcome has probability 1 / 2^K in one tail, or 2 * (1 / 2^K) two-sided
    min_p_k5 = 2.0 * (0.5 ** 5)   # 2 * 1/32 = 0.0625
    min_p_k10 = 2.0 * (0.5 ** 10) # 2 * 1/1024 = 0.001953125

    return {
        "k5_min_two_sided_p": min_p_k5,
        "k10_min_two_sided_p": min_p_k10,
        "is_k5_p_less_than_001_possible": False,
        "is_k10_p_less_than_001_possible": True,
        "proof_statement": (
            "For a two-sided Wilcoxon signed-rank test with sample size K=5, the minimum possible "
            "p-value under maximal rank discordance (W=0 or W=15) is exactly 2 * (1/2)^5 = 0.0625 > 0.01. "
            "Therefore, the manuscript claim in Sec. XXV-C that p < 0.01 across all benchmarks with K=5 runs "
            "is mathematically impossible. In ARES v2, evaluations are executed across K=10 independent random "
            "seeds, where the minimum possible two-sided p-value is 2 * (1/2)^10 = 0.001953125 < 0.01."
        ),
    }


def compute_comprehensive_significance(
    ares_seed_scores: List[float],
    baseline_seed_scores: List[float],
    y_test: np.ndarray,
    ares_preds: np.ndarray,
    baseline_preds: np.ndarray,
    metric_name: str = "macro_f1",
) -> Dict[str, Any]:
    """Runs the full statistical battery between ARES and a baseline model."""
    ares_seeds = np.asarray(ares_seed_scores, dtype=float)
    base_seeds = np.asarray(baseline_seed_scores, dtype=float)

    if len(ares_seeds) != len(base_seeds):
        raise ValueError("Seed score arrays must have equal length.")

    K = len(ares_seeds)

    # 1. Seed-level tests
    diff = ares_seeds - base_seeds
    if np.all(diff == 0):
        t_stat, t_p_val = 0.0, 1.0
        w_stat, w_p_val = 0.0, 1.0
        dz, dz_ci_low, dz_ci_high = 0.0, 0.0, 0.0
    else:
        t_stat, t_p_val = paired_t(ares_seeds, base_seeds)
        w_stat, w_p_val = wilcoxon_exact(ares_seeds, base_seeds)
        sd = np.std(diff, ddof=1)
        if sd == 0 or np.isnan(sd):
            dz = 0.0
            dz_ci_low, dz_ci_high = 0.0, 0.0
        else:
            dz = float(np.mean(diff) / sd)
            se_dz = math.sqrt((1.0 / K) + (dz ** 2 / (2.0 * K)))
            dz_ci_low = float(dz - 1.96 * se_dz)
            dz_ci_high = float(dz + 1.96 * se_dz)

    # 2. Instance-level tests
    def eval_metric(yt, yp):
        classes = np.unique(yt)
        if len(classes) == 0:
            return 0.0
        f1_sum = 0.0
        for c in classes:
            tp = np.count_nonzero((yt == c) & (yp == c))
            fp = np.count_nonzero((yt != c) & (yp == c))
            fn = np.count_nonzero((yt == c) & (yp != c))
            denom = 2 * tp + fp + fn
            if denom > 0:
                f1_sum += (2.0 * tp) / denom
        return float(f1_sum / len(classes))

    delta_boot, ci_l, ci_u, boot_p = paired_bootstrap_delta(
        eval_metric,
        y_test,
        baseline_preds,
        ares_preds,
        B=10000,
        seed=42,
    )

    correct_ares = (ares_preds == y_test)
    correct_base = (baseline_preds == y_test)
    n_disc, mcnemar_p = mcnemar_exact(correct_ares, correct_base)

    return {
        "metric_name": metric_name,
        "K_seeds": K,
        "mean_ares": float(np.mean(ares_seeds)),
        "std_ares": float(np.std(ares_seeds, ddof=1)),
        "mean_baseline": float(np.mean(base_seeds)),
        "std_baseline": float(np.std(base_seeds, ddof=1)),
        "seed_delta_mean": float(np.mean(ares_seeds - base_seeds)),
        # Seed level statistics
        "paired_t_statistic": float(t_stat),
        "paired_t_p_value": float(t_p_val),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_exact_p_value": float(w_p_val),
        "cohens_dz": float(dz),
        "cohens_dz_ci_lower": float(dz_ci_low),
        "cohens_dz_ci_upper": float(dz_ci_high),
        # Instance level statistics
        "bootstrap_delta": float(delta_boot),
        "bootstrap_ci_lower": float(ci_l),
        "bootstrap_ci_upper": float(ci_u),
        "bootstrap_p_value": float(boot_p),
        "mcnemar_discordant_pairs": int(n_disc),
        "mcnemar_exact_p_value": float(mcnemar_p),
        "audit_m1_resolution": prove_wilcoxon_k5_impossibility(),
    }


def apply_holm_bonferroni_battery(
    test_results_list: List[Dict[str, Any]],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Applies step-down Holm-Bonferroni multiple testing adjustment across a family of comparisons."""
    raw_p_values = [res["wilcoxon_exact_p_value"] for res in test_results_list]
    adj_p_values, reject_flags = holm_bonferroni(raw_p_values, alpha=alpha)

    for i, res in enumerate(test_results_list):
        res["wilcoxon_holm_adjusted_p_value"] = float(adj_p_values[i])
        res["statistically_significant_holm"] = bool(reject_flags[i])

    return test_results_list
