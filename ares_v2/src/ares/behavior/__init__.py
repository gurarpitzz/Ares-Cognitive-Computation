"""ARES Behavioral Analysis, Bot Detection, and Social Coordination (Section VII)."""

from ares.behavior.classifier import MultimodalBehaviorClassifier
from ares.behavior.graph import SocialGraphExtractor
from ares.behavior.profile import (
    ProfileFeatureExtractor,
    TwiBotProfileAdapter,
    compute_followers_to_friends_ratio,
    compute_tweet_frequency,
)
from ares.behavior.stylometry import (
    AuthorshipAnomalyOCSVM,
    StylometryExtractor,
    compute_punctuation_variance,
    compute_yules_k,
)
from ares.behavior.temporal import (
    TemporalHMMAuditor,
    compute_autocorrelation_peak,
    compute_burstiness_and_cv,
    compute_inter_post_intervals,
    compute_interval_entropy,
)

__all__ = [
    "compute_yules_k",
    "compute_punctuation_variance",
    "StylometryExtractor",
    "AuthorshipAnomalyOCSVM",
    "compute_inter_post_intervals",
    "compute_interval_entropy",
    "compute_burstiness_and_cv",
    "compute_autocorrelation_peak",
    "TemporalHMMAuditor",
    "SocialGraphExtractor",
    "compute_followers_to_friends_ratio",
    "compute_tweet_frequency",
    "ProfileFeatureExtractor",
    "TwiBotProfileAdapter",
    "MultimodalBehaviorClassifier",
]
