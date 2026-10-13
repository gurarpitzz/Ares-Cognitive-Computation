"""Unit and integrity tests for ARES Multimodal Track-Level Fusion and Ablation Module."""

import math
import numpy as np
import pandas as pd
import pytest
import torch

from ares.fusion.assemble import TrackFeatureAssembler, TRACK_DEFINITIONS
from ares.fusion.late import (
    CalibratedLLRFusion,
    GradientBoostedFusion,
    AppendixAMLPFusion,
    LateFusionPipeline,
)
from ares.fusion.e2e import (
    MultimodalE2EFusionModel,
    E2ECompositeLoss,
)
from ares.fusion.ablation import TrackAblationHarness
from ares.fusion.shapley import ExactShapleyAttributor
from ares.fusion.significance import (
    prove_wilcoxon_k5_impossibility,
    compute_comprehensive_significance,
    apply_holm_bonferroni_battery,
)


def test_track_feature_assembly_and_oof():
    """Verify that feature assembly enforces strict out-of-fold training (Audit Red Flag RF1)."""
    assembler = TrackFeatureAssembler(cache_dir="data/cache/fusion", seed=42)
    df = assembler.assemble_track_dataset("T1", n_samples=300)

    assert "item_id" in df.columns
    assert "split" in df.columns
    assert "label" in df.columns
    assert "is_out_of_fold" in df.columns

    train_rows = df[df["split"] == "train"]
    assert len(train_rows) > 0
    # Audit RF1 assertion: All train rows must be generated out-of-fold
    assert train_rows["is_out_of_fold"].all(), "Audit Red Flag RF1: Train rows must be marked out_of_fold"

    # Module columns check
    for mod in TRACK_DEFINITIONS["T1"]["modules"]:
        assert f"score_{mod}" in df.columns
        assert f"llr_{mod}" in df.columns
        assert f"has_{mod}" in df.columns


def test_late_fusion_heads_and_chow_abstention():
    """Verify late fusion engines and Chow cost-asymmetric abstention (Resolves Audit Issue I-11)."""
    rng = np.random.default_rng(42)
    n = 200
    X_llr = rng.normal(0, 1.5, size=(n, 4))
    y = (X_llr[:, 0] + X_llr[:, 1] > 0).astype(int)

    # 1. Calibrated LLR
    llr_model = CalibratedLLRFusion(seed=42)
    llr_model.fit(X_llr, y, feature_names=["m1", "m2", "m3", "m4"])
    probs = llr_model.predict_proba(X_llr)
    assert probs.shape == (n, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert len(llr_model.weights) == 4

    # 2. Gradient Boosted Fusion
    gbdt_model = GradientBoostedFusion(seed=42)
    gbdt_model.fit(X_llr, y)
    probs_gbdt = gbdt_model.predict_proba(X_llr)
    assert probs_gbdt.shape == (n, 2)

    # 3. Appendix-A MLP Head
    mlp_model = AppendixAMLPFusion(in_features=4, n_classes=2, epochs=5, seed=42)
    mlp_model.fit(X_llr, y)
    probs_mlp = mlp_model.predict_proba(X_llr)
    assert probs_mlp.shape == (n, 2)

    # 4. Chow Abstention Rule
    pipeline = LateFusionPipeline(method="calibrated_llr", c_fp=1.0, c_fn=1.0, c_a=0.20, seed=42)
    pipeline.fit(X_llr, y)
    abstain_res = pipeline.decide_with_abstention(X_llr)
    assert "decisions" in abstain_res
    assert "abstention_rate" in abstain_res
    assert 0.0 <= abstain_res["abstention_rate"] <= 1.0
    valid_decisions = {"PREDICT_0", "PREDICT_1", "ABSTAIN"}
    for d in abstain_res["decisions"]:
        assert d in valid_decisions


def test_e2e_composite_loss_and_vat():
    """Verify Appendix A + Eq. 14 composite loss with embedding-space VAT (Claim C069)."""
    torch.manual_seed(42)
    batch_size = 8
    text_dim = 768
    vision_dim = 512

    model = MultimodalE2EFusionModel(text_dim=text_dim, vision_dim=vision_dim)
    criterion = E2ECompositeLoss(
        lambda_cls=1.0,
        lambda_adv=0.3,
        lambda_reg=1e-4,
        eps=0.02,
        class_weights=(0.7, 1.3),
    )

    e_text = torch.randn(batch_size, text_dim)
    e_vis = torch.randn(batch_size, vision_dim)
    targets = torch.randint(0, 2, (batch_size,))

    loss, metrics = criterion(model, e_text, e_vis, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0
    assert "loss_cls" in metrics
    assert "loss_adv" in metrics
    assert "loss_reg" in metrics
    assert metrics["loss_adv"] >= 0.0
    assert metrics["loss_reg"] >= 0.0

    # Test backward pass
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None


def test_shapley_exact_efficiency_axiom():
    """Verify that exact Shapley values satisfy the efficiency axiom strictly (Audit Red Flag RF5)."""
    assembler = TrackFeatureAssembler(seed=42)
    df = assembler.assemble_track_dataset("T3", n_samples=300)
    active_mods = ["text", "vision", "metadata"]

    attributor = ExactShapleyAttributor(method="calibrated_llr", seed=42, n_bootstraps=50)
    res = attributor.compute_exact_shapley(df, active_mods)

    assert res["efficiency_axiom_verified"], (
        f"Shapley efficiency axiom violated: sum={res['sum_shapley']} vs gain={res['gain_full_over_empty']} "
        f"diff={res['efficiency_difference']}"
    )
    assert len(res["attributions"]) == 3
    for mod in active_mods:
        attr = res["attributions"][mod]
        assert "shapley_value" in attr
        assert "ci_lower" in attr
        assert "ci_upper" in attr
        assert "normalized_share" in attr


def test_second_order_interaction_classification():
    """Verify 2x2 factorial interaction calculation and sub-additivity resolution (Claim C055)."""
    assembler = TrackFeatureAssembler(seed=42)
    df = assembler.assemble_track_dataset("T1", n_samples=300)
    active_mods = ["text", "logic", "temporal", "geo"]

    harness = TrackAblationHarness(method="calibrated_llr", seed=42, n_bootstraps=50)
    res = harness.run_second_order_interaction(df, active_mods, "temporal", "geo")

    assert "interaction_effect" in res
    assert "classification" in res
    assert res["classification"] in {"Super-additive", "Sub-additive", "No detectable interaction"}
    # Verify interaction formula: Delta_{AB} - (Delta_A + Delta_B)
    expected_interaction = res["delta_joint_ab"] - res["sum_individual_deltas"]
    assert np.isclose(res["interaction_effect"], expected_interaction, atol=1e-5)


def test_wilcoxon_k5_vs_k10_mathematical_proof():
    """Verify mathematical proof that K=5 Wilcoxon p < 0.01 was impossible (Audit Issue M-1, Claim C051)."""
    proof = prove_wilcoxon_k5_impossibility()
    assert proof["k5_min_two_sided_p"] == 0.0625
    assert proof["k10_min_two_sided_p"] == 2.0 / 1024.0
    assert not proof["is_k5_p_less_than_001_possible"]
    assert proof["is_k10_p_less_than_001_possible"]
