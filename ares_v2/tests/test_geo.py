"""Unit and integration tests for ARES Geopolitical & Contextual Intelligence Module (Milestone M10)."""

import math
import pytest
import numpy as np

from ares.geo.ner import NamedEntityRecognizer, regenerate_table_iv
from ares.geo.topic import TopicClassifier, regenerate_table_v
from ares.geo.geotag import Gazetteer, Geotagger, haversine_km, regenerate_table_vi
from ares.geo.wikidata_link import WikidataEntityLinker
from ares.geo.gdelt import GDELTCorroborator, run_jk_showcase_trace
from ares.geo.factbook import FactbookContextManager
from ares.geo.risk import GeopoliticalRiskScorer, evaluate_risk_proxy_validation
from ares.geo.cache import ContextFeatureExtractor


def test_haversine_distance_accuracy():
    """Verify haversine formula against known geographic ground truth."""
    # Paris (48.8566 N, 2.3522 E) to London (51.5074 N, -0.1278 W) is ~343 km
    d = haversine_km(48.8566, 2.3522, 51.5074, -0.1278)
    assert 340.0 <= d <= 350.0

    # Distance from a point to itself must be 0
    d_zero = haversine_km(35.6895, 139.6917, 35.6895, 139.6917)
    assert abs(d_zero) < 1e-6


def test_ner_extraction():
    """Verify named entity recognition on sample sentences."""
    ner = NamedEntityRecognizer()
    ents = ner.extract_entities("Elon Musk visited the United Nations in New York City.")
    assert len(ents) >= 2

    ent_texts = [e.text for e in ents]
    assert any("Elon Musk" in t for t in ent_texts)
    assert any("New York City" in t for t in ent_texts)

    locations = ner.extract_locations("Flooding in Tokyo and California.")
    assert len(locations) >= 1


def test_table_iv_regeneration():
    """Verify regeneration of Table IV for Claim C007."""
    rows = regenerate_table_iv()
    assert len(rows) >= 3
    for r in rows:
        assert "Entity Type" in r
        assert "Extracted Entity" in r
        assert "Original Sentence" in r

    extracted_entities = [r["Extracted Entity"] for r in rows]
    assert any("Elon Musk" in e for e in extracted_entities)
    assert any("New York City" in e for e in extracted_entities)
    assert any("Tesla" in e for e in extracted_entities)


def test_topic_classification():
    """Verify topic classifier mapping and confidence."""
    classifier = TopicClassifier(candidate_labels=["Economy", "Technology", "Science", "Health", "Politics"])
    pred = classifier.classify("The economy is facing inflation due to the pandemic.")
    assert pred.predicted_topic in ("Economy", "Health")
    assert 0.0 <= pred.confidence <= 1.0
    assert sum(pred.all_scores.values()) > 0.95


def test_table_v_regeneration():
    """Verify regeneration of Table V for Claim C008."""
    rows = regenerate_table_v()
    assert len(rows) == 3
    for r in rows:
        assert "Text Sample" in r
        assert "Predicted Topic" in r
        assert "Confidence" in r
        assert r["Confidence"] > 0.30


def test_gazetteer_resolution():
    """Verify GeoNames gazetteer resolves major cities and regions."""
    gaz = Gazetteer()
    gaz.load()

    nyc = gaz.resolve_toponym("New York City")
    assert nyc is not None
    assert nyc.country_code == "US"
    assert abs(nyc.latitude - 40.71) < 0.5
    assert abs(nyc.longitude - (-74.00)) < 0.5

    tokyo = gaz.resolve_toponym("Tokyo")
    assert tokyo is not None
    assert tokyo.country_code == "JP"
    assert abs(tokyo.latitude - 35.69) < 0.5

    calif = gaz.resolve_toponym("California")
    assert calif is not None
    assert calif.country_code == "US"


def test_table_vi_regeneration():
    """Verify regeneration of Table VI for Claim C009."""
    rows = regenerate_table_vi()
    assert len(rows) == 3
    for r in rows:
        assert "Input Sentence" in r
        assert "Resolved Location" in r
        loc_str = r["Resolved Location"]
        assert any(k in loc_str for k in ("United States", "USA", "Japan", "Tokyo", "California", "New York"))


def test_wikidata_linking():
    """Verify Wikidata entity linking with Q-ID resolution and prominence score."""
    linker = WikidataEntityLinker()
    linked = linker.link_entity("Elon Musk", context_text="Elon Musk is the CEO of Tesla.")
    assert linked is not None
    assert linked.qid == "Q317521"
    assert linked.prominence_score > 0.80

    linked_who = linker.link_entity("World Health Organization")
    assert linked_who is not None
    assert linked_who.qid == "Q7809"
    assert linked_who.prominence_score >= 0.90


def test_gdelt_strict_temporal_non_leakage():
    """Verify Audit Red Flag RF1: events added after claim date are quarantined."""
    corroborator = GDELTCorroborator()

    # Query with an ancient claim date: no modern events should match
    res = corroborator.corroborate_event(
        event_type="blast",
        location_name="Jammu and Kashmir",
        claim_date_str="2020-01-01",
        window_days=3,
    )
    # The 2026-02-01 event MUST NOT be leaked into the 2020 claim
    assert res.matched_events_count == 0
    assert res.corroboration_score == 0.0
    assert res.strict_leakage_passed is True


def test_jk_showcase_trace():
    """Verify the Jammu & Kashmir showcase event corroboration trace."""
    trace = run_jk_showcase_trace()
    assert trace["matched_events"] >= 1
    assert trace["corroboration_score"] > 0.0
    assert "PASS" in trace["temporal_horizon_check"]
    assert "Jammu and Kashmir" in trace["extracted_location"]


def test_factbook_context():
    """Verify CIA World Factbook context retrieval."""
    fb = FactbookContextManager()
    us_ctx = fb.get_country_context("US")
    assert us_ctx.country_name == "United States"
    assert us_ctx.nuclear_state is True
    assert us_ctx.instability_index < 0.30

    ua_ctx = fb.get_country_context("UA")
    assert ua_ctx.active_conflict_zone is True
    assert ua_ctx.instability_index > 0.80


def test_risk_scoring_equations():
    """Verify mathematical formulations of I_e, E_f, P_g, rho from Section IX-B."""
    scorer = GeopoliticalRiskScorer(w1=0.55, w2=0.45, lambda_thresh=0.65, gamma_thresh=0.60)

    # 1. Event Impact Threshold: I_e = w1 * Sev + w2 * Sig
    ie, esc_ie = scorer.compute_event_impact(severity=0.8, geopolitical_significance=0.7)
    expected_ie = 0.55 * 0.8 + 0.45 * 0.7  # 0.44 + 0.315 = 0.755
    assert abs(ie - expected_ie) < 1e-4
    assert esc_ie is True  # 0.755 > 0.65

    # 2. Entity Involvement Factor: E_f = sum(w_i * C_i)
    ef = scorer.compute_entity_involvement([0.95, 0.85], mention_counts=[1, 1])
    assert abs(ef - 0.90) < 1e-4

    # 3. Conflict Propagation: P_g = sum(a_k * S_k * R_k)
    pg, esc_pg = scorer.compute_conflict_propagation([{"severity": 0.9, "instability_index": 0.8, "a_k": 1.0}])
    expected_pg = 0.9 * 0.8  # 0.72
    assert abs(pg - expected_pg) < 1e-4
    assert esc_pg is True  # 0.72 > 0.60

    # 4. Composite Threat Index
    breakdown = scorer.compute_composite_risk(
        severity=0.8,
        geopolitical_significance=0.7,
        entity_prominences=[0.95, 0.85],
        conflict_signals=[{"severity": 0.9, "instability_index": 0.8, "a_k": 1.0}],
    )
    expected_rho = 0.35 * expected_ie + 0.30 * 0.90 + 0.35 * expected_pg
    assert abs(breakdown.composite_threat_rho - expected_rho) < 1e-4
    assert breakdown.overall_escalated is True


def test_risk_proxy_validation():
    """Verify hypothesis H10.4: risk scores correlate with empirical proxy (Claim C083)."""
    val = evaluate_risk_proxy_validation(n_samples=100)
    assert val["spearman_rho"] > 0.0
    assert val["spearman_pvalue"] < 0.05
    assert val["h10_4_passed"] is True
    assert "REVISED" in val["ledger_claim_C083_status"]


def test_context_feature_extraction():
    """Verify unified context feature vector generation for M11."""
    extractor = ContextFeatureExtractor()
    feat = extractor.extract_features(
        claim_id="test_001",
        text="Breaking: Flooding in New York City prompts state of emergency.",
        claim_date="2023-09-29",
    )
    assert feat.claim_id == "test_001"
    assert feat.has_location is True
    assert feat.country_code == "US"
    assert feat.gdelt_corroboration > 0.0

    vec = feat.to_feature_vector()
    assert len(vec) == 11
    assert all(not math.isnan(x) for x in vec)
