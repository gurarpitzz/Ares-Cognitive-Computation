"""Unit tests for the ARES statistical library."""

import numpy as np
import pytest
from sklearn.metrics import f1_score
from ares.common.stats import (
    bootstrap_ci,
    cohens_dz,
    ece,
    holm_bonferroni,
    macro_f1,
    macro_f1_independent,
    mcnemar_exact,
    paired_bootstrap_delta,
    paired_t,
    wilcoxon_exact,
)


def test_wilcoxon_exact_minimum_p():
    """Verify exact theoretical minimum p-values for Wilcoxon signed-rank test.
    
    For K=5 pairs, min exact two-sided p is 2*(1/2^5) = 2/32 = 0.0625.
    For K=10 pairs, min exact two-sided p is 2*(1/2^10) = 2/1024 = 0.001953125.
    """
    # K=5: maximal rank differences
    a5 = np.zeros(5)
    b5 = np.arange(1, 6, dtype=float)
    _, p5 = wilcoxon_exact(a5, b5)
    assert np.isclose(p5, 0.0625, atol=1e-5), f"Expected 0.0625 for K=5, got {p5}"

    # K=10: maximal rank differences
    a10 = np.zeros(10)
    b10 = np.arange(1, 11, dtype=float)
    _, p10 = wilcoxon_exact(a10, b10)
    assert np.isclose(p10, 0.001953125, atol=1e-6), f"Expected 0.001953125 for K=10, got {p10}"


def test_handwritten_f1_matches_sklearn():
    """Verify that independent hand-written macro-F1 matches scikit-learn bit-for-bit across random trials."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = 200
        n_classes = rng.integers(2, 6)
        y_true = rng.integers(0, n_classes, size=n)
        y_pred = rng.integers(0, n_classes, size=n)
        
        f1_sk = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_ours = macro_f1_independent(y_true, y_pred)
        
        assert np.isclose(f1_sk, f1_ours, atol=1e-12), f"Discrepancy: sklearn={f1_sk}, ours={f1_ours}"


def test_bootstrap_ci_coverage():
    """Verify bootstrap CI on known mean covers true mean."""
    rng = np.random.default_rng(101)
    data = rng.normal(loc=10.0, scale=2.0, size=500)
    
    def mean_fn(yt, yp):
        return float(np.mean(yt))
        
    lower, upper = bootstrap_ci(mean_fn, data, data, B=2000, seed=42)
    assert lower < 10.0 < upper
    assert (upper - lower) < 1.0


def test_paired_bootstrap_delta():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=300)
    pred_a = y_true.copy()
    # introduce errors into a
    pred_a[:50] = 1 - pred_a[:50]
    # pred_b is worse
    pred_b = pred_a.copy()
    pred_b[50:100] = 1 - pred_b[50:100]
    
    def acc_fn(yt, yp):
        return float(np.mean(yt == yp))
        
    delta, lower, upper, p_val = paired_bootstrap_delta(acc_fn, y_true, pred_a, pred_b, B=1000, seed=42)
    # B is worse than A, so delta = B - A should be negative
    assert delta < 0
    assert upper < 0
    assert p_val < 0.05


def test_mcnemar_exact():
    # A correct on 80, B correct on 95, discordant: b=5, c=20
    correct_a = np.array([True]*80 + [False]*20)
    correct_b = np.array([True]*75 + [False]*5 + [True]*20)
    
    n_disc, p_val = mcnemar_exact(correct_a, correct_b)
    assert n_disc == 25
    assert p_val < 0.05


def test_holm_bonferroni():
    raw_p = [0.001, 0.01, 0.04, 0.10]
    adj_p, reject = holm_bonferroni(raw_p, alpha=0.05)
    
    # 0.001 * 4 = 0.004 -> reject
    # 0.01 * 3 = 0.03 -> reject
    # 0.04 * 2 = 0.08 -> do not reject
    # 0.10 * 1 = 0.10 -> do not reject
    assert reject == [True, True, False, False]
    assert adj_p[0] <= adj_p[1] <= adj_p[2] <= adj_p[3]


def test_ece_perfect_vs_uncalibrated():
    # Perfectly calibrated: probs 0.9 with 90% accuracy
    labels = np.array([1]*90 + [0]*10)
    probs = np.full((100, 2), [0.1, 0.9])
    err = ece(probs, labels, n_bins=10)
    assert err < 0.05
