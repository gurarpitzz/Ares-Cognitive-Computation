"""
Unit Tests for M12 Robustness Suite.
"""

import numpy as np
import pytest

from ares.robustness.text_attacks import (
    TextAttackHarness,
    deepwordbug_perturb,
    homoglyph_perturb,
    textfooler_perturb,
)
from ares.robustness.cross_domain import compute_proxy_a_distance


def test_textfooler_perturb_replaces_words():
    text = "The government confirmed 20 covid deaths in the hospital."
    perturbed, mods, rate = textfooler_perturb(text, budget_frac=0.30, rng=np.random.RandomState(42))
    assert mods >= 1
    assert rate > 0.0
    assert perturbed != text


def test_deepwordbug_perturb_modifies_characters():
    text = "Emergency services deployed immediately across the flooded district."
    perturbed, mods, rate = deepwordbug_perturb(text, budget_frac=0.20, rng=np.random.RandomState(42))
    assert mods >= 1
    assert perturbed != text


def test_homoglyph_perturb_inserts_unicode():
    text = "Official reports state that peace talks concluded yesterday."
    perturbed, mods, rate = homoglyph_perturb(text, budget_frac=0.20, rng=np.random.RandomState(42))
    assert mods >= 1
    assert perturbed != text


def test_proxy_a_distance_identical_distributions():
    rng = np.random.RandomState(42)
    # Identical distributions should yield low proxy A-distance (near 0)
    X1 = rng.normal(0, 1, size=(200, 20))
    X2 = rng.normal(0, 1, size=(200, 20))
    d_A = compute_proxy_a_distance(X1, X2, n_splits=3, seed=42)
    assert 0.0 <= d_A <= 0.60


def test_proxy_a_distance_separated_distributions():
    rng = np.random.RandomState(42)
    # Fully separable distributions should yield high proxy A-distance (near 2.0)
    X1 = rng.normal(5.0, 0.5, size=(200, 20))
    X2 = rng.normal(-5.0, 0.5, size=(200, 20))
    d_A = compute_proxy_a_distance(X1, X2, n_splits=3, seed=42)
    assert d_A >= 1.60
