"""ARES Vision Module: Visual Deepfake Detection & Forensics (Section X, Appendix A)."""

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
from ares.vision.audit_table11 import (
    calculate_f1,
    calculate_precision,
    solve_class_ratio_for_f1,
    run_table11_audit,
)
from ares.vision.data import (
    get_train_transforms,
    get_eval_transforms,
    FaceFrameDataset,
    BalancedClassBatchSampler,
    visualize_batch_sample,
)

__all__ = [
    "DeepfakeDetector",
    "mean_pooling",
    "topk_entropy_pooling",
    "find_operating_threshold",
    "compute_video_metrics",
    "bootstrap_video_metrics",
    "compute_azimuthal_average_fft",
    "compute_eye_aspect_ratio",
    "compute_mouth_aspect_ratio",
    "extract_video_landmark_jitter",
    "ForensicFeatureClassifier",
    "LateFusionDetector",
    "calculate_f1",
    "calculate_precision",
    "solve_class_ratio_for_f1",
    "run_table11_audit",
    "get_train_transforms",
    "get_eval_transforms",
    "FaceFrameDataset",
    "BalancedClassBatchSampler",
    "visualize_batch_sample",
]
