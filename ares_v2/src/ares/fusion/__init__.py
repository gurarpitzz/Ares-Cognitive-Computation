"""ARES Multimodal Track-Level Fusion and Ablation Module."""

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
    E2ETrainer,
)
from ares.fusion.ablation import TrackAblationHarness
from ares.fusion.shapley import ExactShapleyAttributor
from ares.fusion.significance import (
    prove_wilcoxon_k5_impossibility,
    compute_comprehensive_significance,
    apply_holm_bonferroni_battery,
)

__all__ = [
    "TrackFeatureAssembler",
    "TRACK_DEFINITIONS",
    "CalibratedLLRFusion",
    "GradientBoostedFusion",
    "AppendixAMLPFusion",
    "LateFusionPipeline",
    "MultimodalE2EFusionModel",
    "E2ECompositeLoss",
    "E2ETrainer",
    "TrackAblationHarness",
    "ExactShapleyAttributor",
    "prove_wilcoxon_k5_impossibility",
    "compute_comprehensive_significance",
    "apply_holm_bonferroni_battery",
]
