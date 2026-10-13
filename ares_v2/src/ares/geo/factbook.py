"""CIA World Factbook Context module for ARES Geopolitical & Contextual Intelligence.

Provides country-level demographic, political governance, military capability,
and regional instability indices (R_k) for downstream risk escalation (Section IX).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard country intelligence profiles based on CIA World Factbook & Fragile States Index
CURATED_FACTBOOK_PROFILES: Dict[str, Dict[str, Any]] = {
    "US": {
        "country_name": "United States",
        "iso2": "US",
        "iso3": "USA",
        "population": 339000000,
        "government_type": "constitutional federal republic",
        "gdp_per_capita_usd": 76300,
        "military_expenditure_pct_gdp": 3.5,
        "instability_index": 0.15,  # R_k: low baseline instability
        "nuclear_state": True,
        "active_conflict_zone": False,
    },
    "IN": {
        "country_name": "India",
        "iso2": "IN",
        "iso3": "IND",
        "population": 1428000000,
        "government_type": "federal parliamentary republic",
        "gdp_per_capita_usd": 8400,
        "military_expenditure_pct_gdp": 2.4,
        "instability_index": 0.45,  # R_k: moderate regional tension in specific borders (e.g. LOC)
        "nuclear_state": True,
        "active_conflict_zone": False,
    },
    "JP": {
        "country_name": "Japan",
        "iso2": "JP",
        "iso3": "JPN",
        "population": 124000000,
        "government_type": "parliamentary constitutional monarchy",
        "gdp_per_capita_usd": 42100,
        "military_expenditure_pct_gdp": 1.1,
        "instability_index": 0.10,
        "nuclear_state": False,
        "active_conflict_zone": False,
    },
    "CN": {
        "country_name": "China",
        "iso2": "CN",
        "iso3": "CHN",
        "population": 1412000000,
        "government_type": "communist party-led state",
        "gdp_per_capita_usd": 18200,
        "military_expenditure_pct_gdp": 1.7,
        "instability_index": 0.35,
        "nuclear_state": True,
        "active_conflict_zone": False,
    },
    "RU": {
        "country_name": "Russia",
        "iso2": "RU",
        "iso3": "RUS",
        "population": 144000000,
        "government_type": "semi-presidential federation",
        "gdp_per_capita_usd": 29850,
        "military_expenditure_pct_gdp": 4.1,
        "instability_index": 0.85,  # High geopolitical friction / active war
        "nuclear_state": True,
        "active_conflict_zone": True,
    },
    "UA": {
        "country_name": "Ukraine",
        "iso2": "UA",
        "iso3": "UKR",
        "population": 38000000,
        "government_type": "semi-presidential republic",
        "gdp_per_capita_usd": 12600,
        "military_expenditure_pct_gdp": 33.5,
        "instability_index": 0.95,  # Active full-scale kinetic conflict
        "nuclear_state": False,
        "active_conflict_zone": True,
    },
    "FR": {
        "country_name": "France",
        "iso2": "FR",
        "iso3": "FRA",
        "population": 68000000,
        "government_type": "semi-presidential republic",
        "gdp_per_capita_usd": 45000,
        "military_expenditure_pct_gdp": 1.9,
        "instability_index": 0.18,
        "nuclear_state": True,
        "active_conflict_zone": False,
    },
    "DE": {
        "country_name": "Germany",
        "iso2": "DE",
        "iso3": "DEU",
        "population": 84000000,
        "government_type": "federal parliamentary republic",
        "gdp_per_capita_usd": 54000,
        "military_expenditure_pct_gdp": 1.4,
        "instability_index": 0.12,
        "nuclear_state": False,
        "active_conflict_zone": False,
    },
    "IL": {
        "country_name": "Israel",
        "iso2": "IL",
        "iso3": "ISR",
        "population": 9800000,
        "government_type": "parliamentary democracy",
        "gdp_per_capita_usd": 49000,
        "military_expenditure_pct_gdp": 4.5,
        "instability_index": 0.90,  # Active conflict zone
        "nuclear_state": True,
        "active_conflict_zone": True,
    },
    "PS": {
        "country_name": "Palestine",
        "iso2": "PS",
        "iso3": "PSE",
        "population": 5200000,
        "government_type": "semi-presidential authority",
        "gdp_per_capita_usd": 5600,
        "military_expenditure_pct_gdp": 0.0,
        "instability_index": 0.98,  # Extreme conflict/humanitarian emergency
        "nuclear_state": False,
        "active_conflict_zone": True,
    },
}

GLOBAL_AVERAGE_PROFILE: Dict[str, Any] = {
    "country_name": "Global Average / Unknown",
    "iso2": "XX",
    "iso3": "XXX",
    "population": 50000000,
    "government_type": "constitutional republic",
    "gdp_per_capita_usd": 15000,
    "military_expenditure_pct_gdp": 2.0,
    "instability_index": 0.40,
    "nuclear_state": False,
    "active_conflict_zone": False,
}


@dataclass
class CountryFactbookContext:
    """Represents geopolitical context derived from CIA World Factbook."""
    country_name: str
    iso2: str
    iso3: str
    population: int
    government_type: str
    gdp_per_capita_usd: float
    military_expenditure_pct_gdp: float
    instability_index: float
    nuclear_state: bool
    active_conflict_zone: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "country_name": self.country_name,
            "iso2": self.iso2,
            "iso3": self.iso3,
            "population": self.population,
            "government_type": self.government_type,
            "gdp_per_capita_usd": self.gdp_per_capita_usd,
            "military_expenditure_pct_gdp": self.military_expenditure_pct_gdp,
            "instability_index": round(self.instability_index, 2),
            "nuclear_state": self.nuclear_state,
            "active_conflict_zone": self.active_conflict_zone,
        }


class FactbookContextManager:
    """Manages country factbook intelligence with offline caching (RF4)."""

    def __init__(
        self,
        snapshot_file: str = "milestones/M10_geo_context/cache/factbook_profiles.json",
    ):
        self.snapshot_file = snapshot_file
        self._profiles: Dict[str, Dict[str, Any]] = dict(CURATED_FACTBOOK_PROFILES)
        self._load_snapshot()

    def _load_snapshot(self):
        if os.path.exists(self.snapshot_file):
            try:
                with open(self.snapshot_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._profiles.update(data)
            except Exception as e:
                logger.warning("Failed to load factbook snapshot: %s", e)
        else:
            os.makedirs(os.path.dirname(self.snapshot_file), exist_ok=True)
            try:
                with open(self.snapshot_file, "w", encoding="utf-8") as f:
                    json.dump(self._profiles, f, indent=2)
            except Exception as e:
                logger.warning("Failed to save default factbook snapshot: %s", e)

    def get_country_context(self, country_code_or_name: str) -> CountryFactbookContext:
        """Retrieve country intelligence profile by ISO-2 code or common country name."""
        key = country_code_or_name.strip().upper()
        if key in self._profiles:
            return CountryFactbookContext(**self._profiles[key])

        # Search by country name (case-insensitive)
        name_lower = country_code_or_name.strip().lower()
        for prof in self._profiles.values():
            if prof["country_name"].lower() == name_lower:
                return CountryFactbookContext(**prof)

        return CountryFactbookContext(**GLOBAL_AVERAGE_PROFILE)
