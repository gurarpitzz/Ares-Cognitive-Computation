"""Unit Tests for Claim Drift, Mutation Detection, and Temporal Memory (M06).

Verifies assumption tests, theoretical CDFs, Lipschitz estimation,
temporal claim memory non-leakage (RF1), and claim chain regression.
"""

import numpy as np
import pytest

from ares.drift.assumptions import assess_isotropy, assess_gaussianity_pca, assess_norm_independence
from ares.drift.theory_comparison import compute_theoretical_cosine_cdf, evaluate_theory_vs_empirical_roc
from ares.drift.lipschitz import estimate_lipschitz_quantiles
from ares.drift.memory import TemporalClaimMemory
from ares.drift.chains import compute_chain_drift_trajectory, regress_verification_log_odds_drift


def test_isotropy_synthetic():
    rng = np.random.default_rng(42)
    # 1. Perfectly isotropic noise in d=50
    iso_noise = rng.normal(0, 1.0, size=(500, 50))
    res_iso = assess_isotropy(iso_noise)
    assert not res_iso["isotropy_rejected"], "Isotropic Gaussian should not be rejected"
    assert res_iso["anisotropy_ratio"] > 0.50
    
    # 2. Highly anisotropic noise (first component has 100x variance)
    aniso_noise = iso_noise.copy()
    aniso_noise[:, 0] *= 100.0
    res_aniso = assess_isotropy(aniso_noise)
    assert res_aniso["isotropy_rejected"], "Anisotropic noise must be rejected"
    assert res_aniso["condition_number"] > 100.0


def test_theoretical_cosine_cdf():
    tau_grid = np.linspace(-0.8, 0.8, 50)
    cdf = compute_theoretical_cosine_cdf(tau_grid, kappa=0.1, dimension=128)
    
    assert len(cdf) == 50
    assert np.all((cdf >= 0.0) & (cdf <= 1.0))
    # Must be monotonically non-decreasing
    for i in range(len(cdf) - 1):
        assert cdf[i+1] >= cdf[i] - 1e-6


def test_lipschitz_quantiles():
    rng = np.random.default_rng(42)
    n = 100
    e1 = rng.normal(0, 1, size=(n, 64))
    e2 = e1 + rng.normal(0, 0.5, size=(n, 64))
    scores = rng.uniform(1.0, 5.0, size=n)
    
    res = estimate_lipschitz_quantiles(e1, e2, scores)
    q = res["quantiles"]
    assert q["L_50"] <= q["L_75"] <= q["L_90"] <= q["L_95"] <= q["L_99"]


def test_temporal_claim_memory_non_leakage_rf1():
    """Verify Red Flag RF1: Strict temporal non-leakage in memory store."""
    memory = TemporalClaimMemory(theta_match=0.80)
    
    dim = 32
    rng = np.random.default_rng(42)
    v1 = rng.normal(0, 1, dim)
    v2 = rng.normal(0, 1, dim)
    v3 = rng.normal(0, 1, dim)
    
    # Add 3 claims at t=10, t=20, t=30
    memory.add_claim("c1", "Claim at t=10", v1, verdict=1.0, timestamp=10)
    memory.add_claim("c2", "Claim at t=20", v2, verdict=-1.0, timestamp=20)
    memory.add_claim("c3", "Claim at t=30", v3, verdict=1.0, timestamp=30)
    
    # Query at t=25: MUST NEVER return c3 (timestamp 30)
    q_emb = v3.copy()  # even if query is identical to c3
    res_25 = memory.query(q_emb, query_timestamp=25, top_k=5)
    
    assert res_25["has_history"]
    matched_ids = [m["item_id"] for m in res_25["top_k_matches"]]
    assert "c3" not in matched_ids, "RF1 Violation: Claim from future timestamp returned!"
    assert set(matched_ids).issubset({"c1", "c2"})
    
    # Query before all memories (t=5): MUST return empty history
    res_5 = memory.query(q_emb, query_timestamp=5, top_k=5)
    assert not res_5["has_history"]
    assert len(res_5["top_k_matches"]) == 0


def test_chain_drift_regression():
    claims = ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]
    rng = np.random.default_rng(42)
    embs = rng.normal(0, 1, size=(4, 32))
    mutation_flags = [0, 0, 1, 1]
    
    df_traj = compute_chain_drift_trajectory(claims, embs, mutation_flags)
    assert len(df_traj) == 4
    assert df_traj["cumulative_drift"].iloc[-1] >= 0.0
    
    # Test log-odds regression
    log_odds = np.array([2.5, 2.3, -1.5, -2.0])
    res_reg = regress_verification_log_odds_drift(log_odds, np.array(mutation_flags))
    assert res_reg["kappa"] > 0.0
    assert "ci95_low" in res_reg
