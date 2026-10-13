"""Wikidata Entity Linking module for ARES Geopolitical & Contextual Intelligence.

Maps extracted entities to Wikidata items (Q-IDs) with semantic embedding re-ranking,
supporting deterministic offline snapshot replay (RF4) and entity prominence scoring for E_f.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Curated core entity knowledge graph snapshot for offline deterministic replay
CURATED_WIKIDATA_ENTITIES: Dict[str, Dict[str, Any]] = {
    "elon musk": {
        "qid": "Q317521",
        "label": "Elon Musk",
        "description": "South African-born American entrepreneur, business magnate and investor",
        "prominence_score": 0.95,
        "entity_type": "Q5",  # Human
    },
    "tesla": {
        "qid": "Q478214",
        "label": "Tesla, Inc.",
        "description": "American automotive and clean energy company",
        "prominence_score": 0.85,
        "entity_type": "Q4830453",  # Business
    },
    "new york city": {
        "qid": "Q60",
        "label": "New York City",
        "description": "most populous city in the United States",
        "prominence_score": 0.90,
        "entity_type": "Q515",  # City
    },
    "tokyo": {
        "qid": "Q1490",
        "label": "Tokyo",
        "description": "capital and most populous prefecture of Japan",
        "prominence_score": 0.90,
        "entity_type": "Q515",
    },
    "california": {
        "qid": "Q99",
        "label": "California",
        "description": "state of the United States of America",
        "prominence_score": 0.88,
        "entity_type": "Q35657",  # State of the US
    },
    "united nations": {
        "qid": "Q1065",
        "label": "United Nations",
        "description": "intergovernmental organization founded in 1945",
        "prominence_score": 1.00,
        "entity_type": "Q1065",
    },
    "world health organization": {
        "qid": "Q7809",
        "label": "World Health Organization",
        "description": "specialized agency of the United Nations responsible for international public health",
        "prominence_score": 0.98,
        "entity_type": "Q7809",
    },
    "who": {
        "qid": "Q7809",
        "label": "World Health Organization",
        "description": "specialized agency of the United Nations responsible for international public health",
        "prominence_score": 0.98,
        "entity_type": "Q7809",
    },
    "narendra modi": {
        "qid": "Q1058",
        "label": "Narendra Modi",
        "description": "Prime Minister of India since 2014",
        "prominence_score": 0.95,
        "entity_type": "Q5",
    },
    "jammu and kashmir": {
        "qid": "Q1180",
        "label": "Jammu and Kashmir",
        "description": "union territory of India",
        "prominence_score": 0.85,
        "entity_type": "Q6256",
    },
    "vladimir putin": {
        "qid": "Q7747",
        "label": "Vladimir Putin",
        "description": "President of Russia",
        "prominence_score": 0.98,
        "entity_type": "Q5",
    },
    "volodymyr zelenskyy": {
        "qid": "Q55643485",
        "label": "Volodymyr Zelenskyy",
        "description": "President of Ukraine",
        "prominence_score": 0.95,
        "entity_type": "Q5",
    },
    "joe biden": {
        "qid": "Q6279",
        "label": "Joe Biden",
        "description": "46th President of the United States",
        "prominence_score": 0.98,
        "entity_type": "Q5",
    },
    "donald trump": {
        "qid": "Q22686",
        "label": "Donald Trump",
        "description": "45th and 47th President of the United States",
        "prominence_score": 0.98,
        "entity_type": "Q5",
    },
}


@dataclass
class LinkedEntity:
    """Represents a disambiguated Wikidata entity."""
    entity_text: str
    qid: str
    label: str
    description: str
    similarity_score: float
    prominence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_text": self.entity_text,
            "qid": self.qid,
            "label": self.label,
            "description": self.description,
            "similarity_score": round(self.similarity_score, 4),
            "prominence_score": round(self.prominence_score, 4),
        }


class WikidataEntityLinker:
    """Links entity strings to Wikidata items using embeddings and offline snapshots."""

    def __init__(
        self,
        cache_file: str = "milestones/M10_geo_context/cache/wikidata_cache.json",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.cache_file = cache_file
        self.embed_model_name = embed_model_name
        self._embedder = None
        self._cache: Dict[str, Dict[str, Any]] = dict(CURATED_WIKIDATA_ENTITIES)
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._cache.update(data)
            except Exception as e:
                logger.warning("Could not load wikidata cache from %s: %s", self.cache_file, e)

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning("Could not save wikidata cache to %s: %s", self.cache_file, e)

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            # Search in local HF hub cache first for offline execution
            pattern = os.path.expanduser(f"~/.cache/huggingface/hub/models--{self.embed_model_name.replace('/', '--')}/snapshots/*")
            snaps = glob.glob(pattern)
            if snaps:
                self._embedder = SentenceTransformer(snaps[0])
            else:
                self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def search_candidates(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve candidate entities from snapshot or live search."""
        key = query.strip().lower()
        if key in self._cache:
            cand = self._cache[key]
            return [cand]

        # Live search fallback if online
        try:
            import requests
            url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "format": "json",
                "limit": 5,
            }
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                results = resp.json().get("search", [])
                candidates = []
                for item in results:
                    candidates.append({
                        "qid": item.get("id"),
                        "label": item.get("label", ""),
                        "description": item.get("description", ""),
                        "prominence_score": 0.50,
                    })
                if candidates:
                    self._cache[key] = candidates[0]
                    self._save_cache()
                return candidates
        except Exception as e:
            logger.debug("Live Wikidata search failed for '%s': %s", query, e)

        return []

    def link_entity(self, entity_text: str, context_text: Optional[str] = None) -> Optional[LinkedEntity]:
        """Link an entity mention to its best Wikidata candidate using semantic re-ranking."""
        candidates = self.search_candidates(entity_text)
        if not candidates:
            return None

        if len(candidates) == 1 or not context_text:
            top_cand = candidates[0]
            return LinkedEntity(
                entity_text=entity_text,
                qid=top_cand["qid"],
                label=top_cand["label"],
                description=top_cand.get("description", ""),
                similarity_score=1.0,
                prominence_score=top_cand.get("prominence_score", 0.50),
            )

        embedder = self._get_embedder()
        context_emb = embedder.encode(context_text)

        best_cand = candidates[0]
        best_sim = -1.0

        for cand in candidates:
            cand_desc = f"{cand['label']}: {cand.get('description', '')}"
            cand_emb = embedder.encode(cand_desc)

            sim = float(np.dot(context_emb, cand_emb) / (np.linalg.norm(context_emb) * np.linalg.norm(cand_emb) + 1e-9))
            if sim > best_sim:
                best_sim = sim
                best_cand = cand

        return LinkedEntity(
            entity_text=entity_text,
            qid=best_cand["qid"],
            label=best_cand["label"],
            description=best_cand.get("description", ""),
            similarity_score=best_sim,
            prominence_score=best_cand.get("prominence_score", 0.50),
        )
