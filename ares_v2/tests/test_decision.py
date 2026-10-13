"""Unit Tests for ARES Decision Layer (M05).

Verifies calibration, LLR fusion, heuristic voting, Chow abstention,
simulated HITL, and the C062 latency arithmetic proof.
"""

import numpy as np
import pytest

from ares.decision.calibration import PlattScaler, TemperatureScaler, compute_ece, compute_brier_score
from ares.decision.llr import NaiveBayesLLRFusion, LearnedLLRFusion
from ares.decision.voting import HeuristicWeightedVoting
from ares.decision.abstain import ChowAbstention, compute_risk_coverage_curve
from ares.decision.hitl_sim import SimulatedAnalyst, sweep_static_hitl
from ares.decision.latency_model import compute_expected_latency


def test_calibration_reduces_ece():
    rng = np.random.default_rng(42)
    # Generate uncalibrated, overconfident probabilities
    y_true = rng.integers(0, 2, size=200)
    raw_probs = np.where(y_true == 1, rng.uniform(0.4, 0.95, 200), rng.uniform(0.05, 0.6, 200))
    # Artificial overconfidence
    overconf_probs = np.where(raw_probs > 0.5, 0.99, 0.01)
    
    uncal_ece = compute_ece(overconf_probs, y_true)
    
    scaler = PlattScaler()
    scaler.fit(overconf_probs, y_true)
    cal_probs = scaler.transform(overconf_probs)
    
    cal_ece = compute_ece(cal_probs, y_true)
    assert cal_ece < uncal_ece, f"Calibrated ECE ({cal_ece:.4f}) should be lower than uncalibrated ({uncal_ece:.4f})"


def test_llr_fusion_and_correlations():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=150)
    
    # 3 correlated signal sources
    s1 = np.where(y_true == 1, rng.normal(1.0, 0.8, 150), rng.normal(-1.0, 0.8, 150))
    s2 = s1 * 0.9 + rng.normal(0, 0.2, 150) # strongly correlated with s1
    s3 = np.where(y_true == 1, rng.normal(0.5, 1.0, 150), rng.normal(-0.5, 1.0, 150))
    
    signals = np.column_stack([s1, s2, s3])
    
    nb = NaiveBayesLLRFusion()
    nb_post = nb.predict_proba(signals)
    assert nb_post.shape == (150, 2)
    assert np.all((nb_post >= 0) & (nb_post <= 1.0))
    
    learned = LearnedLLRFusion(C=1.0)
    learned.fit(signals, y_true)
    learned_post = learned.predict_proba(signals)
    assert learned_post.shape == (150, 2)
    
    # Due to correlation between s1 and s2, learned weights should adjust
    assert learned.weights.shape == (3,)


def test_heuristic_weighted_voting():
    engine = HeuristicWeightedVoting(tau=0.65)
    
    # 3 models predicting class 1 with high confidence
    predictions = [(1, 0.9), (1, 0.8), (1, 0.85)]
    res = engine.vote(predictions)
    assert res["winner"] == 1
    assert not res["escalate"]
    
    # Conflicting models resulting in low max score
    predictions_conflict = [(0, 0.50), (1, 0.50)]
    res_c = engine.vote(predictions_conflict)
    assert res_c["escalate"], "Low maximum voting score should trigger escalation"


def test_chow_abstention_and_aurc():
    # Theorem C Chow abstention
    chow = ChowAbstention(c_a=0.2, c_fp=1.0, c_fn=1.0)
    
    # Balanced costs => symmetric thresholds around 0
    assert np.isclose(chow.tau_1, -chow.tau_0)
    
    # Check decisions
    assert chow.decide(2.5) == "PREDICT_1"
    assert chow.decide(-2.5) == "PREDICT_0"
    assert chow.decide(0.0) == "ABSTAIN"
    
    probs = np.array([0.95, 0.05, 0.52, 0.48, 0.85, 0.15])
    labels = np.array([1, 0, 1, 0, 1, 0])
    curve = compute_risk_coverage_curve(probs, labels)
    assert "aurc" in curve
    assert "selective_accuracy_milestones" in curve


def test_simulated_hitl_monotonicity():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    # Poor model accuracy ~0.60
    probs = np.zeros((100, 2))
    for i in range(100):
        if rng.random() < 0.60:
            probs[i, y_true[i]] = rng.uniform(0.6, 0.9)
            probs[i, 1 - y_true[i]] = 1.0 - probs[i, y_true[i]]
        else:
            probs[i, 1 - y_true[i]] = rng.uniform(0.6, 0.9)
            probs[i, y_true[i]] = 1.0 - probs[i, 1 - y_true[i]]
            
    df = sweep_static_hitl(y_true, probs, escalation_rates=[0.0, 0.10, 0.20, 0.30])
    
    oracle_df = df[df["analyst_type"].str.contains("ORACLE")]
    accs = oracle_df["accuracy"].tolist()
    
    # Accuracy must increase monotonically as more claims are escalated to the oracle
    for i in range(len(accs) - 1):
        assert accs[i+1] >= accs[i]


def test_c062_latency_arithmetic_proof():
    res = compute_expected_latency(t_auto_ms=415.0, t_human_min=18.6, escalation_rate=0.068)
    
    # True expected latency: 76.275s
    assert 76.0 < res["true_expected_latency_s"] < 77.0
    
    # Buggy manuscript reproduction: 0.068 * 18.6 = 1.2648 s
    assert 1.25 < res["manuscript_buggy_reproduction_s"] < 1.27
    
    # Discrepancy factor: true latency is ~60x higher than reported
    assert res["discrepancy_factor"] > 55.0
