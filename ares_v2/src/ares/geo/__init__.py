"""ARES Geopolitical & Contextual Intelligence Module (Milestone M10)."""

from ares.geo.ner import NamedEntityRecognizer, ExtractedEntity, evaluate_conll2003, regenerate_table_iv
from ares.geo.topic import TopicClassifier, TopicPrediction, regenerate_table_v, STANDARD_DOMAINS
from ares.geo.geotag import Gazetteer, Geotagger, GeocodedLocation, haversine_km, evaluate_geowebnews_benchmark, regenerate_table_vi
from ares.geo.wikidata_link import WikidataEntityLinker, LinkedEntity
from ares.geo.gdelt import GDELTCorroborator, GDELTEventRecord, CorroborationResult, run_jk_showcase_trace, evaluate_gdelt_event_discriminability
from ares.geo.factbook import FactbookContextManager, CountryFactbookContext
from ares.geo.risk import GeopoliticalRiskScorer, RiskBreakdown, evaluate_risk_proxy_validation
from ares.geo.cache import ContextFeatureExtractor, ItemContextFeatures

__all__ = [
    "NamedEntityRecognizer",
    "ExtractedEntity",
    "evaluate_conll2003",
    "regenerate_table_iv",
    "TopicClassifier",
    "TopicPrediction",
    "regenerate_table_v",
    "STANDARD_DOMAINS",
    "Gazetteer",
    "Geotagger",
    "GeocodedLocation",
    "haversine_km",
    "evaluate_geowebnews_benchmark",
    "regenerate_table_vi",
    "WikidataEntityLinker",
    "LinkedEntity",
    "GDELTCorroborator",
    "GDELTEventRecord",
    "CorroborationResult",
    "run_jk_showcase_trace",
    "evaluate_gdelt_event_discriminability",
    "FactbookContextManager",
    "CountryFactbookContext",
    "GeopoliticalRiskScorer",
    "RiskBreakdown",
    "evaluate_risk_proxy_validation",
    "ContextFeatureExtractor",
    "ItemContextFeatures",
]
