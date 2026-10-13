"""Unit and Integration Tests for Behavioral Analysis and Bot Detection (Milestone M09)."""

import numpy as np
import pandas as pd
import pytest
import torch

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


class TestStylometry:
    """Tests for stylometric analysis and authorship anomaly detection (Section VII-A)."""

    def test_yules_k_hand_calculation(self):
        # All unique words: N=4, f_1=4 -> sum(i^2 * f_i) = 4 -> K = 10^4 * (4 - 4) / 16 = 0
        unique_text = "the quick brown fox"
        k_unique = compute_yules_k(unique_text)
        assert k_unique == 0.0

        # Highly repetitive: "dog" repeated 4 times: N=4, f_4=1 -> sum = 16 -> K = 10^4 * (16 - 4) / 16 = 7500.0
        rep_text = "dog dog dog dog"
        k_rep = compute_yules_k(rep_text)
        assert np.isclose(k_rep, 7500.0)

    def test_yules_k_empty_or_single_token(self):
        assert compute_yules_k("") == 0.0
        assert compute_yules_k("hello") == 0.0

    def test_punctuation_variance(self):
        text_no_punct = "This is a clean sentence without any punctuation"
        var_zero, _ = compute_punctuation_variance(text_no_punct)
        assert var_zero == 0.0

        text_varied = "First sentence! Second sentence has lots of commas, semicolons; and dashes--really? Third is plain."
        var_pos, mean_p = compute_punctuation_variance(text_varied)
        assert var_pos > 0.0
        assert mean_p > 0.0

    def test_stylometry_extractor_and_ocsvm(self):
        human_texts = [
            "Just enjoying a lovely cup of coffee this morning! The weather is gorgeous.",
            "Can't believe how busy work has been today, feeling quite tired...",
            "Going to the park with friends later this afternoon. Hope it stays sunny!",
            "Loved the latest episode of that podcast, really thoughtful analysis.",
            "Cooked dinner for the family tonight, pasta turned out great.",
        ]
        bot_texts = [
            "CLICK HERE: http://spam.co/offer Win $1000 NOW! BUY BUY BUY",
            "HOT DEAL: http://promo.biz/discount 50% OFF TODAY ONLY!!!",
            "FREE BITCOIN GIVEAWAY: register at http://crypto.fake/gift instantly!",
        ]

        extractor = StylometryExtractor(max_char_ngrams=20)
        extractor.fit(human_texts)
        X_human = extractor.transform(human_texts)
        X_bot = extractor.transform(bot_texts)

        assert X_human.shape[0] == 5
        assert X_bot.shape[0] == 3

        # OCSVM trained strictly on humans
        ocsvm = AuthorshipAnomalyOCSVM(nu=0.1)
        ocsvm.fit(X_human)

        scores_human = ocsvm.score_samples(X_human)
        scores_bot = ocsvm.score_samples(X_bot)

        # Bot anomaly scores should on average exceed human anomaly scores
        assert np.mean(scores_bot) > np.mean(scores_human)


class TestTemporal:
    """Tests for temporal posting dynamics and HMM LLR (Section VII-B)."""

    def test_inter_post_intervals(self):
        ts = np.array([1000, 1050, 1100, 1200])
        ints = compute_inter_post_intervals(ts)
        assert np.allclose(ints, [50, 50, 100])

        assert len(compute_inter_post_intervals(np.array([500]))) == 0
        assert len(compute_inter_post_intervals(np.array([]))) == 0

    def test_burstiness_and_cv(self):
        # Perfectly regular intervals: sigma = 0 -> B = -1, CV = 0
        regular_ints = np.array([60.0, 60.0, 60.0, 60.0, 60.0])
        b, cv = compute_burstiness_and_cv(regular_ints)
        assert b == -1.0
        assert cv == 0.0

        # Bursty intervals: high variance
        bursty_ints = np.array([1.0, 1.0, 2.0, 5000.0, 1.0, 1.0, 8000.0])
        b_burst, cv_burst = compute_burstiness_and_cv(bursty_ints)
        assert b_burst > 0.0
        assert cv_burst > 1.0

    def test_interval_entropy(self):
        regular_ints = np.array([60.0, 60.0, 60.0, 60.0])
        assert compute_interval_entropy(regular_ints) == 0.0

        diverse_ints = np.array([1.0, 10.0, 60.0, 3600.0, 86400.0])
        ent = compute_interval_entropy(diverse_ints)
        assert ent > 0.0

    def test_autocorrelation_peak(self):
        regular_ints = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        corr, lag = compute_autocorrelation_peak(regular_ints)
        assert corr == 1.0

    def test_temporal_hmm_auditor(self):
        rng = np.random.default_rng(42)
        # Bursty human: exponential distribution
        human_seqs = [rng.exponential(scale=3600.0, size=25) for _ in range(15)]
        # Periodic bot: tight gaussian around 300 seconds
        bot_seqs = [np.clip(rng.normal(loc=300.0, scale=2.0, size=25), 1.0, None) for _ in range(15)]

        auditor = TemporalHMMAuditor(n_components=2, n_iter=20, random_state=42)
        auditor.fit(human_seqs, bot_seqs)

        test_bot = np.clip(rng.normal(loc=300.0, scale=2.0, size=20), 1.0, None)
        test_human = rng.exponential(scale=3600.0, size=20)

        llr_bot = auditor.compute_log_likelihood_ratio(test_bot)
        llr_human = auditor.compute_log_likelihood_ratio(test_human)

        assert llr_bot > llr_human


class TestGraph:
    """Tests for social graph topology and coordination metrics (Section VII-C)."""

    def test_graph_centrality_and_louvain(self):
        # 4-node graph: 0 connects to 1, 1 connects to 2, 2 connects to 0 (triangle), 3 connects to 0
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 0, 0]])
        train_indices = np.array([0, 1, 2])
        train_labels = np.array([1, 1, 0])

        extractor = SocialGraphExtractor(inductive=False, max_target_nodes=4)
        extractor.fit(edge_index, train_indices, train_labels)

        feats = extractor.transform(np.array([0, 1, 2, 3]))
        assert feats.shape == (4, 9)

        # In-degree of node 0: connected from 2 and 3 -> in_deg = 2
        assert feats[0, 0] == 2.0
        # Out-degree of node 0: connected to 1 -> out_deg = 1
        assert feats[0, 1] == 1.0

    def test_inductive_mode_isolation(self):
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 0, 0]])
        train_indices = np.array([0, 1, 2])
        train_labels = np.array([0, 0, 1])

        # Inductive: only edges between train nodes (0, 1, 2) are kept
        extractor = SocialGraphExtractor(inductive=True, max_target_nodes=4)
        extractor.fit(edge_index, train_indices, train_labels)

        # Node 3 is not in training nodes, should have degree 0 in inductive graph
        feats = extractor.transform(np.array([3]))
        assert feats[0, 0] == 0.0  # in-degree


class TestProfile:
    """Tests for profile metadata features (Section VII-B/C)."""

    def test_followers_ratio_and_tweet_rate(self):
        assert np.isclose(compute_followers_to_friends_ratio(1000, 999), 1.0)
        assert np.isclose(compute_tweet_frequency(365, 365), 1.0)
        assert compute_tweet_frequency(100, 0) == 100.0  # Safe division

    def test_profile_extractor_dataframe(self):
        df = pd.DataFrame([
            {"followers_count": 1000, "friends_count": 200, "statuses_count": 5000, "favourites_count": 100, "account_age_days": 500, "verified": True},
            {"followers_count": 5, "friends_count": 1500, "statuses_count": 25000, "favourites_count": 1, "account_age_days": 10, "verified": False},
        ])
        ext = ProfileFeatureExtractor()
        feats = ext.extract_from_dataframe(df)
        assert feats.shape == (2, 9)
        ext.fit(feats)
        norm = ext.transform(feats)
        assert norm.shape == (2, 9)

    def test_twibot_profile_adapter(self):
        num_prop = np.random.randn(10, 5)
        cat_prop = np.ones((10, 3))
        train_idx = np.arange(7)

        adapter = TwiBotProfileAdapter()
        norm, _ = adapter.fit_transform(num_prop, cat_prop, train_idx)
        assert norm.shape == (10, 9)


class TestMultimodalClassifier:
    """Tests for fused multimodal behavioral classifier (Section VII)."""

    def test_classifier_fit_and_predict(self):
        n_samples = 50
        y = np.random.randint(0, 2, size=n_samples)

        s_feats = np.random.randn(n_samples, 4)
        t_feats = np.random.randn(n_samples, 5)
        p_feats = np.random.randn(n_samples, 9)
        g_feats = np.random.randn(n_samples, 9)

        clf = MultimodalBehaviorClassifier(classifier_type="logistic_regression")
        clf.fit(y, stylometry_feats=s_feats, temporal_feats=t_feats, profile_feats=p_feats, graph_feats=g_feats)

        probs = clf.predict_proba(stylometry_feats=s_feats, temporal_feats=t_feats, profile_feats=p_feats, graph_feats=g_feats)
        preds = clf.predict(stylometry_feats=s_feats, temporal_feats=t_feats, profile_feats=p_feats, graph_feats=g_feats)

        assert probs.shape == (n_samples, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert len(preds) == n_samples

    def test_feature_ablation_mode(self):
        n_samples = 30
        y = np.random.randint(0, 2, size=n_samples)
        p_feats = np.random.randn(n_samples, 9)

        # Profile-only classifier
        clf = MultimodalBehaviorClassifier(classifier_type="logistic_regression", active_families=["profile"])
        clf.fit(y, profile_feats=p_feats)

        probs = clf.predict_proba(profile_feats=p_feats)
        assert probs.shape == (n_samples, 2)
