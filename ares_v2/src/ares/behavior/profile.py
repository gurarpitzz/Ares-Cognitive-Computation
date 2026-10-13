"""Profile Metadata Feature Extraction (Section VII-B/C).

Implements:
1. Followers-to-friends ratio: \\rho = \\text{followers} / (\\text{friends} + 1).
2. Logarithmic scale transforms: \\log(followers + 1), \\log(friends + 1), \\log(statuses + 1).
3. Tweet production frequency: statuses / account_age_days.
4. Categorical metadata indicators (verified status, default profile, geo-enabled).
5. Standardized scikit-learn compatible ProfileFeatureExtractor supporting both
   raw dictionary/tabular data and pre-encoded benchmark tensors (TwiBot-20).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_followers_to_friends_ratio(followers: float, friends: float) -> float:
    r"""Compute followers-to-friends ratio \rho = followers / (friends + 1).
    
    Args:
        followers: Number of followers.
        friends: Number of friends/followings.
    Returns:
        ratio: Non-negative ratio.
    """
    return float(max(0.0, followers) / (max(0.0, friends) + 1.0))


def compute_tweet_frequency(statuses_count: float, account_age_days: float) -> float:
    """Compute average tweets posted per day."""
    return float(max(0.0, statuses_count) / max(1.0, account_age_days))


class ProfileFeatureExtractor:
    """Extracts and normalizes account profile metadata features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = [
            "log_followers",
            "log_friends",
            "log_statuses",
            "log_favourites",
            "follower_friend_ratio",
            "tweet_frequency",
            "is_verified",
            "default_profile",
            "geo_enabled",
        ]

    def extract_from_dict(self, profile: Dict[str, Any]) -> np.ndarray:
        """Extract profile feature vector from a single user profile dictionary."""
        followers = float(profile.get("followers_count", 0) or 0)
        friends = float(profile.get("friends_count", 0) or 0)
        statuses = float(profile.get("statuses_count", 0) or 0)
        favourites = float(profile.get("favourites_count", 0) or 0)
        age = float(profile.get("account_age_days", 365) or 365)

        log_followers = float(np.log1p(max(0.0, followers)))
        log_friends = float(np.log1p(max(0.0, friends)))
        log_statuses = float(np.log1p(max(0.0, statuses)))
        log_fav = float(np.log1p(max(0.0, favourites)))
        ratio = compute_followers_to_friends_ratio(followers, friends)
        freq = float(profile.get("average_tweets_per_day", 0.0) or compute_tweet_frequency(statuses, age))

        verified = 1.0 if bool(profile.get("verified", False)) else 0.0
        default_prof = 1.0 if bool(profile.get("default_profile", False)) else 0.0
        geo = 1.0 if bool(profile.get("geo_enabled", False)) else 0.0

        return np.array([
            log_followers,
            log_friends,
            log_statuses,
            log_fav,
            ratio,
            freq,
            verified,
            default_prof,
            geo,
        ], dtype=float)

    def extract_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Extract profile feature matrix from a pandas DataFrame with vectorized operations."""
        followers = pd.to_numeric(df.get("followers_count", 0), errors="coerce").fillna(0).clip(lower=0).values
        friends = pd.to_numeric(df.get("friends_count", 0), errors="coerce").fillna(0).clip(lower=0).values
        statuses = pd.to_numeric(df.get("statuses_count", 0), errors="coerce").fillna(0).clip(lower=0).values
        favourites = pd.to_numeric(df.get("favourites_count", 0), errors="coerce").fillna(0).clip(lower=0).values
        age = pd.to_numeric(df.get("account_age_days", 365), errors="coerce").fillna(365).clip(lower=1).values

        log_followers = np.log1p(followers)
        log_friends = np.log1p(friends)
        log_statuses = np.log1p(statuses)
        log_fav = np.log1p(favourites)
        ratio = followers / (friends + 1.0)

        if "average_tweets_per_day" in df.columns:
            freq = pd.to_numeric(df["average_tweets_per_day"], errors="coerce").fillna(0).clip(lower=0).values
        else:
            freq = statuses / np.maximum(1.0, age)

        if "is_verified" in df.columns:
            verified = df["is_verified"].astype(bool).astype(float).values
        elif "verified" in df.columns:
            verified = df["verified"].astype(bool).astype(float).values
        else:
            verified = np.zeros(len(df), dtype=float)

        if "default_profile" in df.columns:
            default_prof = df["default_profile"].astype(bool).astype(float).values
        else:
            default_prof = np.zeros(len(df), dtype=float)

        if "geo_enabled" in df.columns:
            geo = df["geo_enabled"].astype(bool).astype(float).values
        else:
            geo = np.zeros(len(df), dtype=float)

        return np.column_stack([
            log_followers,
            log_friends,
            log_statuses,
            log_fav,
            ratio,
            freq,
            verified,
            default_prof,
            geo,
        ]).astype(float)

    def fit(self, X_train: np.ndarray) -> "ProfileFeatureExtractor":
        """Fit standard scaler strictly on training split features (RF1)."""
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform profile features using fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("ProfileFeatureExtractor must be fitted before transforming")
        return self.scaler.transform(X)


class TwiBotProfileAdapter:
    """Adapts TwiBot-20 preprocessed num_properties and cat_properties tensors into features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(
        self,
        num_prop: np.ndarray,
        cat_prop: np.ndarray,
        train_indices: np.ndarray,
    ) -> Tuple[np.ndarray, "TwiBotProfileAdapter"]:
        """Fit scaler on training slice and transform full tensor into feature matrix."""
        # num_prop shape: (N, 5), cat_prop shape: (N, 3)
        # Combine num and cat, plus ratio features:
        # num_prop[0] = followers, num_prop[1] = friends
        ratio_feat = (num_prop[:, 0] - num_prop[:, 1]).reshape(-1, 1)
        raw_combined = np.hstack([num_prop, cat_prop, ratio_feat])

        # Fit strictly on train indices (RF1)
        self.scaler.fit(raw_combined[train_indices])
        self.is_fitted = True
        normalized = self.scaler.transform(raw_combined)
        return normalized, self

    def transform(self, num_prop: np.ndarray, cat_prop: np.ndarray) -> np.ndarray:
        """Transform num and cat properties using fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("TwiBotProfileAdapter must be fitted first")
        ratio_feat = (num_prop[:, 0] - num_prop[:, 1]).reshape(-1, 1)
        raw_combined = np.hstack([num_prop, cat_prop, ratio_feat])
        return self.scaler.transform(raw_combined)
