"""Context Feature Cache module for ARES Geopolitical & Contextual Intelligence.

Extracts and caches item-level geopolitical context features (NER, geotags, GDELT, Factbook, rho, domain)
for downstream multimodal fusion in M11 (Claims C043, C047, C053, C054).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ItemContextFeatures:
    """Consolidated context feature representation for a single claim item."""
    claim_id: str
    text: str
    n_entities: int
    has_location: bool
    resolved_lat: float
    resolved_lon: float
    country_code: str
    instability_index: float
    gdelt_corroboration: float
    event_impact_Ie: float
    entity_involvement_Ef: float
    conflict_propagation_Pg: float
    composite_threat_rho: float
    predicted_domain: str
    domain_confidence: float

    def to_feature_vector(self) -> List[float]:
        """Convert scalar indicators into a numerical vector for fusion models."""
        return [
            float(self.n_entities),
            1.0 if self.has_location else 0.0,
            self.resolved_lat / 90.0,
            self.resolved_lon / 180.0,
            self.instability_index,
            self.gdelt_corroboration,
            self.event_impact_Ie,
            self.entity_involvement_Ef,
            self.conflict_propagation_Pg,
            self.composite_threat_rho,
            self.domain_confidence,
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "n_entities": self.n_entities,
            "has_location": self.has_location,
            "resolved_lat": round(self.resolved_lat, 4),
            "resolved_lon": round(self.resolved_lon, 4),
            "country_code": self.country_code,
            "instability_index": round(self.instability_index, 2),
            "gdelt_corroboration": round(self.gdelt_corroboration, 4),
            "event_impact_Ie": round(self.event_impact_Ie, 4),
            "entity_involvement_Ef": round(self.entity_involvement_Ef, 4),
            "conflict_propagation_Pg": round(self.conflict_propagation_Pg, 4),
            "composite_threat_rho": round(self.composite_threat_rho, 4),
            "predicted_domain": self.predicted_domain,
            "domain_confidence": round(self.domain_confidence, 4),
        }


class ContextFeatureExtractor:
    """Unified extractor synthesizing outputs from all M10 submodules."""

    def __init__(self):
        from ares.geo.factbook import FactbookContextManager
        from ares.geo.gdelt import GDELTCorroborator
        from ares.geo.geotag import Gazetteer, Geotagger
        from ares.geo.ner import NamedEntityRecognizer
        from ares.geo.risk import GeopoliticalRiskScorer
        from ares.geo.topic import TopicClassifier
        from ares.geo.wikidata_link import WikidataEntityLinker

        self.ner = NamedEntityRecognizer()
        self.gazetteer = Gazetteer()
        self.geotagger = Geotagger(self.gazetteer)
        self.topic_classifier = TopicClassifier()
        self.entity_linker = WikidataEntityLinker()
        self.gdelt = GDELTCorroborator()
        self.factbook = FactbookContextManager()
        self.risk_scorer = GeopoliticalRiskScorer()

    def extract_features(
        self,
        claim_id: str,
        text: str,
        claim_date: Optional[str] = None,
        dataset_source: Optional[str] = None,
    ) -> ItemContextFeatures:
        """Extract the full suite of geopolitical context features for an item."""
        # 1. NER
        ents = self.ner.extract_entities(text)
        n_ents = len(ents)

        # 2. Geotagging
        locations = self.geotagger.resolve_text_locations(text)
        has_loc = len(locations) > 0
        if has_loc:
            prim_loc = locations[0]
            lat = prim_loc.latitude
            lon = prim_loc.longitude
            cc = prim_loc.country_code
        else:
            lat = 0.0
            lon = 0.0
            cc = "XX"

        # 3. Topic / Domain
        topic_pred = self.topic_classifier.classify(text, dataset_source=dataset_source)

        # 4. Country Context
        fb_ctx = self.factbook.get_country_context(cc)

        # 5. GDELT Event Corroboration
        gdelt_score = 0.0
        if has_loc and claim_date:
            ev_res = self.gdelt.corroborate_event(
                event_type=topic_pred.predicted_topic,
                location_name=locations[0].name,
                claim_date_str=claim_date,
            )
            gdelt_score = ev_res.corroboration_score

        # 6. Entity Prominence
        ent_prominences = []
        for e in ents:
            linked = self.entity_linker.link_entity(e.text, context_text=text)
            if linked:
                ent_prominences.append(linked.prominence_score)
            else:
                ent_prominences.append(0.30)

        # 7. Risk Scores
        sev_proxy = 0.8 if topic_pred.predicted_topic in ("Conflict", "Disaster") else 0.3
        sig_proxy = 0.9 if fb_ctx.nuclear_state or fb_ctx.active_conflict_zone else 0.4
        conf_signals = [{"severity": sev_proxy, "instability_index": fb_ctx.instability_index, "a_k": 1.0}]

        risk_breakdown = self.risk_scorer.compute_composite_risk(
            severity=sev_proxy,
            geopolitical_significance=sig_proxy,
            entity_prominences=ent_prominences,
            conflict_signals=conf_signals,
        )

        return ItemContextFeatures(
            claim_id=claim_id,
            text=text,
            n_entities=n_ents,
            has_location=has_loc,
            resolved_lat=lat,
            resolved_lon=lon,
            country_code=cc,
            instability_index=fb_ctx.instability_index,
            gdelt_corroboration=gdelt_score,
            event_impact_Ie=risk_breakdown.event_impact_Ie,
            entity_involvement_Ef=risk_breakdown.entity_involvement_Ef,
            conflict_propagation_Pg=risk_breakdown.conflict_propagation_Pg,
            composite_threat_rho=risk_breakdown.composite_threat_rho,
            predicted_domain=topic_pred.predicted_topic,
            domain_confidence=topic_pred.confidence,
        )
