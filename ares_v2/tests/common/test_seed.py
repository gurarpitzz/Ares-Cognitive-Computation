"""Unit tests for deterministic seeding."""

import numpy as np
import torch
from ares.common.seed import set_all


def test_seed_determinism_numpy():
    set_all(42)
    a1 = np.random.rand(10)
    set_all(42)
    a2 = np.random.rand(10)
    np.testing.assert_array_equal(a1, a2)


def test_seed_determinism_torch():
    set_all(123)
    t1 = torch.randn(5, 5)
    set_all(123)
    t2 = torch.randn(5, 5)
    assert torch.equal(t1, t2)
