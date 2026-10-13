"""Unit and Regression Tests for Vision Deepfake Module (M07)."""

import numpy as np
import pytest
from PIL import Image
import torch

from ares.vision.audit_table11 import (
    calculate_f1,
    calculate_precision,
    solve_class_ratio_for_f1,
    run_table11_audit,
)
from ares.vision.models import DeepfakeDetector
from ares.vision.aggregation import (
    mean_pooling,
    topk_entropy_pooling,
    find_operating_threshold,
    compute_video_metrics,
    bootstrap_video_metrics,
)
from ares.vision.forensics import (
    compute_azimuthal_average_fft,
    compute_eye_aspect_ratio,
    compute_mouth_aspect_ratio,
    extract_video_landmark_jitter,
    ForensicFeatureClassifier,
    LateFusionDetector,
)
from ares.vision.data import (
    get_train_transforms,
    get_eval_transforms,
    BalancedClassBatchSampler,
)


def test_audit_table11_mathematical_proof():
    """Verify closed-form formula and mathematical resolution of Audit Issue I-14."""
    # Balanced classes (r = 1.0)
    ares_f1_bal = calculate_f1(tpr=0.95, fpr=0.025, r=1.0)
    assert abs(ares_f1_bal - 0.962025) < 1e-4, f"Expected 0.9620, got {ares_f1_bal}"
    
    cnn_f1_bal = calculate_f1(tpr=0.89, fpr=0.08, r=1.0)
    assert abs(cnn_f1_bal - 0.903553) < 1e-4, f"Expected 0.9036, got {cnn_f1_bal}"

    # FaceForensics++ c23 official test split ratio (140 real / 560 fake => r = 0.25)
    ares_f1_ffpp = calculate_f1(tpr=0.95, fpr=0.025, r=0.25)
    assert abs(ares_f1_ffpp - 0.971246) < 1e-4, f"Expected ~0.9712, got {ares_f1_ffpp}"

    # Round-trip root solver check
    target_f1 = 0.92
    solved_r = solve_class_ratio_for_f1(tpr=0.89, fpr=0.08, target_f1=target_f1)
    reconstructed_f1 = calculate_f1(tpr=0.89, fpr=0.08, r=solved_r)
    assert abs(reconstructed_f1 - target_f1) < 1e-5

    # Run full audit
    records = run_table11_audit()
    assert len(records) == 2
    assert records[0].model_name == "ARES (ResNet-50)"
    assert abs(records[0].balanced_discrepancy - (0.97 - 0.962025)) < 1e-4


def test_models_forward_and_feature_extraction():
    """Verify DeepfakeDetector models output valid logits and feature representations."""
    # Test ResNet-50 (untrained backbone to keep unit test fast)
    model = DeepfakeDetector(architecture="resnet50", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    
    logits = model(x)
    assert logits.shape == (2, 1)
    
    probs = model.predict_proba(x)
    assert probs.shape == (2, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    
    features = model.extract_features(x)
    assert features.shape == (2, model.num_features)
    assert model.num_features == 2048


def test_aggregation_operators():
    """Verify video-level frame pooling and validation threshold selection."""
    # Video 1: Consistent fake frames
    v1_frames = [0.85, 0.92, 0.88, 0.95, 0.89]
    assert mean_pooling(v1_frames) == pytest.approx(0.898, abs=1e-3)
    
    # Video 2: Intermittent manipulation
    v2_frames = [0.10, 0.15, 0.90, 0.12, 0.14]
    assert mean_pooling(v2_frames) == pytest.approx(0.282, abs=1e-3)
    top3_entropy = topk_entropy_pooling(v2_frames, k=3)
    assert 0.0 <= top3_entropy <= 1.0

    # Threshold selection on synthetic validation set
    y_true_val = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_score_val = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.5, 0.7, 0.8, 0.9, 0.95])
    
    thresh_youden, stats_y = find_operating_threshold(y_true_val, y_score_val, criterion="youden")
    assert 0.0 < thresh_youden < 1.0
    assert stats_y["tpr"] >= 0.8

    # Video metrics computation
    metrics = compute_video_metrics(y_true_val, y_score_val, threshold=thresh_youden, r=1.0)
    assert metrics["roc_auc"] > 0.8
    assert "stated_ratio_f1" in metrics
    
    # Bootstrap CI
    cis = bootstrap_video_metrics(y_true_val, y_score_val, threshold=thresh_youden, n_boot=200, seed=42)
    assert "roc_auc" in cis
    assert cis["roc_auc"][0] <= cis["roc_auc"][1]


def test_forensics_fft_and_landmarks():
    """Verify FFT power spectrum extraction, landmark jitter, and late fusion."""
    # 1. FFT Radial Profile
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    profile = compute_azimuthal_average_fft(img, num_bins=32)
    assert profile.shape == (32,)
    assert np.all(np.isfinite(profile))

    # 2. Facial Landmark Dynamics
    rng = np.random.default_rng(42)
    seq = [rng.uniform(50, 150, size=(68, 2)) for _ in range(5)]
    jitter_stats = extract_video_landmark_jitter(seq)
    assert "ear_variance" in jitter_stats
    assert "landmark_jitter_mean" in jitter_stats
    assert jitter_stats["landmark_jitter_mean"] >= 0.0

    # 3. Forensic Classifier & Late Fusion
    X_train = rng.normal(size=(20, 38))  # 32 FFT bins + 6 landmark features
    y_train = np.array([0]*10 + [1]*10)
    
    clf = ForensicFeatureClassifier(classifier_type="gbm", seed=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_train)
    assert probs.shape == (20,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Late fusion
    fusion = LateFusionDetector(alpha=0.8)
    cnn_s = np.array([0.9, 0.2, 0.8, 0.1])
    for_s = np.array([0.8, 0.3, 0.7, 0.2])
    fused = fusion.fuse(cnn_s, for_s)
    assert len(fused) == 4
    assert fused[0] == pytest.approx(0.8 * 0.9 + 0.2 * 0.8)


def test_data_transforms_and_sampler():
    """Verify augmentations and balanced class sampler."""
    t_train = get_train_transforms(image_size=224)
    t_eval = get_eval_transforms(image_size=224)
    
    img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    tensor_train = t_train(img)
    tensor_eval = t_eval(img)
    
    assert tensor_train.shape == (3, 224, 224)
    assert tensor_eval.shape == (3, 224, 224)

    # Balanced Sampler: 20 real, 80 fake (4:1 imbalance)
    labels = [0] * 20 + [1] * 80
    sampler = BalancedClassBatchSampler(labels=labels, batch_size=8, seed=42)
    
    batches = list(sampler)
    assert len(batches) > 0
    # Every batch must have exactly 4 real (idx < 20) and 4 fake (idx >= 20)
    for b in batches:
        real_in_batch = sum(1 for idx in b if labels[idx] == 0)
        fake_in_batch = sum(1 for idx in b if labels[idx] == 1)
        assert real_in_batch == 4
        assert fake_in_batch == 4
