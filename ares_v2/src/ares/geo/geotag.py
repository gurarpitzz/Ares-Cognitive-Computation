"""Geotagging & Toponym Resolution module for ARES Geopolitical & Contextual Intelligence.

Implements Eq. (5) in Section V-C:
- Toponym extraction via NER
- GeoNames gazetteer candidate generation and spatial disambiguation
- EXIF GPS extraction from imagery
- Great-circle Haversine distance evaluation (Accuracy@161km on GeoWebNews, H10.2)
- Table VI regeneration (Claim C009)
"""

from __future__ import annotations

import logging
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points on Earth in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


@dataclass
class GeocodedLocation:
    """Represents a resolved geographic location."""
    name: str
    country_code: str
    country_name: str
    latitude: float
    longitude: float
    population: int = 0
    admin1: str = ""
    geoname_id: Optional[str] = None
    confidence: float = 1.0

    def to_formatted_str(self) -> str:
        """Format as: City/Region, Country (Lat°N/S, Lon°E/W)."""
        lat_dir = "N" if self.latitude >= 0 else "S"
        lon_dir = "E" if self.longitude >= 0 else "W"
        return f"{self.name}, {self.country_name} ({abs(self.latitude):.2f}°{lat_dir}, {abs(self.longitude):.2f}°{lon_dir})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "country_code": self.country_code,
            "country_name": self.country_name,
            "latitude": round(self.latitude, 4),
            "longitude": round(self.longitude, 4),
            "population": self.population,
            "admin1": self.admin1,
            "geoname_id": self.geoname_id,
            "formatted": self.to_formatted_str(),
        }


# High-level regional/state coordinates for administrative regions often mentioned in claims
REGIONAL_CENTROIDS: Dict[str, Dict[str, Any]] = {
    "california": {"name": "California", "country_code": "US", "country_name": "USA", "latitude": 36.7783, "longitude": -119.4179, "population": 39000000},
    "texas": {"name": "Texas", "country_code": "US", "country_name": "USA", "latitude": 31.9686, "longitude": -99.9018, "population": 30000000},
    "florida": {"name": "Florida", "country_code": "US", "country_name": "USA", "latitude": 27.6648, "longitude": -81.5158, "population": 22000000},
    "new york state": {"name": "New York", "country_code": "US", "country_name": "USA", "latitude": 43.0000, "longitude": -75.0000, "population": 19500000},
    "jammu and kashmir": {"name": "Jammu and Kashmir", "country_code": "IN", "country_name": "India", "latitude": 33.7782, "longitude": 76.5762, "population": 12500000},
    "kashmir": {"name": "Jammu and Kashmir", "country_code": "IN", "country_name": "India", "latitude": 33.7782, "longitude": 76.5762, "population": 12500000},
    "gaza": {"name": "Gaza Strip", "country_code": "PS", "country_name": "Palestine", "latitude": 31.3547, "longitude": 34.3088, "population": 2100000},
    "crimea": {"name": "Crimea", "country_code": "UA", "country_name": "Ukraine", "latitude": 45.3453, "longitude": 34.4997, "population": 2400000},
    "taiwan": {"name": "Taiwan", "country_code": "TW", "country_name": "Taiwan", "latitude": 23.6978, "longitude": 120.9605, "population": 23500000},
}


class Gazetteer:
    """GeoNames gazetteer indexer with population prior and context disambiguation."""

    def __init__(
        self,
        cities_path: str = "data/raw/geonames/cities15000.txt",
        country_info_path: str = "data/raw/geonames/countryInfo.txt",
    ):
        self.cities_path = cities_path
        self.country_info_path = country_info_path
        self._name_index: Dict[str, List[Dict[str, Any]]] = {}
        self._country_names: Dict[str, str] = {}  # code -> name
        self._country_codes: Dict[str, str] = {}  # name -> code
        self._loaded = False

    def load(self):
        """Load country names and cities gazetteer into memory."""
        if self._loaded:
            return

        # 1. Load Country Info
        if os.path.exists(self.country_info_path):
            with open(self.country_info_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 5:
                        iso2 = parts[0].upper()
                        cname = parts[4]
                        self._country_names[iso2] = cname
                        self._country_codes[cname.lower()] = iso2

        # Standard fallbacks
        self._country_names.setdefault("US", "USA")
        self._country_names.setdefault("GB", "UK")
        self._country_names.setdefault("JP", "Japan")
        self._country_names.setdefault("IN", "India")
        self._country_names.setdefault("CN", "China")
        self._country_names.setdefault("RU", "Russia")
        self._country_names.setdefault("UA", "Ukraine")

        for c, n in list(self._country_names.items()):
            self._country_codes[n.lower()] = c

        # 2. Load Cities15000
        if os.path.exists(self.cities_path):
            with open(self.cities_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 15:
                        continue
                    gid = parts[0]
                    name = parts[1]
                    asciiname = parts[2]
                    altnames = parts[3].split(",") if parts[3] else []
                    lat = float(parts[4])
                    lon = float(parts[5])
                    cc = parts[8].upper()
                    admin1 = parts[10]
                    pop = int(parts[14]) if parts[14].isdigit() else 0

                    cname = self._country_names.get(cc, cc)
                    record = {
                        "geoname_id": gid,
                        "name": name,
                        "country_code": cc,
                        "country_name": cname,
                        "latitude": lat,
                        "longitude": lon,
                        "population": pop,
                        "admin1": admin1,
                    }

                    # Index primary and ascii names
                    for key in {name.lower(), asciiname.lower()}:
                        if key not in self._name_index:
                            self._name_index[key] = []
                        self._name_index[key].append(record)

                    # Index top alternate names
                    for alt in altnames[:5]:
                        alt_clean = alt.strip().lower()
                        if alt_clean and len(alt_clean) > 2:
                            if alt_clean not in self._name_index:
                                self._name_index[alt_clean] = []
                            self._name_index[alt_clean].append(record)

        self._loaded = True
        logger.info("Gazetteer loaded: %d distinct name keys.", len(self._name_index))

    def resolve_toponym(
        self,
        toponym: str,
        context_country: Optional[str] = None,
        context_text: Optional[str] = None,
    ) -> Optional[GeocodedLocation]:
        """Disambiguate a toponym string to a geocoded coordinate pair."""
        if not self._loaded:
            self.load()

        clean_name = toponym.strip().lower()

        # Check regional centroids first
        if clean_name in REGIONAL_CENTROIDS:
            r = REGIONAL_CENTROIDS[clean_name]
            return GeocodedLocation(
                name=r["name"],
                country_code=r["country_code"],
                country_name=r["country_name"],
                latitude=r["latitude"],
                longitude=r["longitude"],
                population=r["population"],
            )

        # Check country name directly
        if clean_name in self._country_codes:
            cc = self._country_codes[clean_name]
            # Return capital or centroid
            candidates = self._name_index.get(clean_name, [])
            if candidates:
                best = max(candidates, key=lambda c: c["population"])
                return GeocodedLocation(**best)

        # Gazetteer candidate lookup
        candidates = self._name_index.get(clean_name, [])
        if not candidates:
            # Try removing common prefixes/suffixes: "City of New York", "New York City" -> "New York"
            if clean_name.endswith(" city"):
                base = clean_name[:-5].strip()
                candidates = self._name_index.get(base, [])
            elif clean_name.startswith("city of "):
                base = clean_name[8:].strip()
                candidates = self._name_index.get(base, [])

        if not candidates:
            return None

        # Disambiguation scoring:
        # Score = log10(pop + 1) + 5.0 (if country match) + 2.0 (if exact name)
        best_cand = None
        best_score = -1.0

        target_cc = context_country.upper() if context_country else None
        if not target_cc and context_text:
            text_lower = context_text.lower()
            for cname_low, cc in self._country_codes.items():
                if len(cname_low) > 3 and cname_low in text_lower:
                    target_cc = cc
                    break

        for cand in candidates:
            pop = cand["population"]
            score = math.log10(pop + 1.0)
            if cand["name"].lower() == clean_name:
                score += 2.0
            if target_cc and cand["country_code"] == target_cc:
                score += 5.0

            if score > best_score:
                best_score = score
                best_cand = cand

        if best_cand:
            return GeocodedLocation(
                name=best_cand["name"],
                country_code=best_cand["country_code"],
                country_name=best_cand["country_name"],
                latitude=best_cand["latitude"],
                longitude=best_cand["longitude"],
                population=best_cand["population"],
                admin1=best_cand.get("admin1", ""),
                geoname_id=best_cand.get("geoname_id"),
            )
        return None


class Geotagger:
    """Full geotagging pipeline integrating NER, gazetteer resolution, and EXIF extraction."""

    def __init__(self, gazetteer: Optional[Gazetteer] = None):
        from ares.geo.ner import NamedEntityRecognizer
        self.ner = NamedEntityRecognizer()
        self.gazetteer = gazetteer or Gazetteer()

    def resolve_text_locations(self, text: str) -> List[GeocodedLocation]:
        """Extract and resolve all location mentions in a text."""
        loc_mentions = self.ner.extract_locations(text)
        results = []
        seen = set()

        for mention in loc_mentions:
            clean = mention.strip().lower()
            if clean in seen or len(clean) < 2:
                continue
            seen.add(clean)

            loc = self.gazetteer.resolve_toponym(mention, context_text=text)
            if loc:
                results.append(loc)

        return results

    def extract_image_exif_gps(self, image_path: str) -> Optional[Tuple[float, float, Optional[str]]]:
        """Extract GPS latitude, longitude, and timestamp from image EXIF metadata."""
        if not os.path.exists(image_path):
            return None

        try:
            import exifread
            with open(image_path, "rb") as f:
                tags = exifread.process_file(f, details=False)

            lat_tag = tags.get("GPS GPSLatitude")
            lat_ref = tags.get("GPS GPSLatitudeRef")
            lon_tag = tags.get("GPS GPSLongitude")
            lon_ref = tags.get("GPS GPSLongitudeRef")
            time_tag = tags.get("EXIF DateTimeOriginal") or tags.get("GPS GPSTimeStamp")

            if lat_tag and lat_ref and lon_tag and lon_ref:
                lat_vals = [float(x.num) / float(x.den) for x in lat_tag.values]
                lon_vals = [float(x.num) / float(x.den) for x in lon_tag.values]

                lat = lat_vals[0] + lat_vals[1] / 60.0 + lat_vals[2] / 3600.0
                lon = lon_vals[0] + lon_vals[1] / 60.0 + lon_vals[2] / 3600.0

                if str(lat_ref.values).upper() in ("S", "SOUTH"):
                    lat = -lat
                if str(lon_ref.values).upper() in ("W", "WEST"):
                    lon = -lon

                ts = str(time_tag.values) if time_tag else None
                return lat, lon, ts
        except Exception as e:
            logger.warning("EXIF extraction failed on %s: %s", image_path, e)

        return None


def evaluate_geowebnews_benchmark(
    gwn_xml_path: str = "data/raw/toponyms/GWN.xml",
    gazetteer: Optional[Gazetteer] = None,
    distance_threshold_km: float = 161.0,
    max_eval: Optional[int] = 500,
) -> Dict[str, Any]:
    """Evaluate toponym resolution accuracy@161km on the GeoWebNews benchmark (Gritta et al., 2018).

    Validates Hypothesis H10.2.
    """
    if gazetteer is None:
        gazetteer = Gazetteer()
        gazetteer.load()

    if not os.path.exists(gwn_xml_path):
        raise FileNotFoundError(f"GeoWebNews benchmark file not found at: {gwn_xml_path}")

    tree = ET.parse(gwn_xml_path)
    root = tree.getroot()

    total_evaluated = 0
    resolved_count = 0
    accurate_161_count = 0
    accurate_50_count = 0
    error_distances: List[float] = []

    for article in root:
        text_elem = article.find("text")
        art_text = text_elem.text if text_elem is not None else ""
        toponyms_elem = article.find("toponyms")
        if toponyms_elem is None:
            continue

        for t in toponyms_elem:
            name_elem = t.find("extractedName")
            lat_elem = t.find("latitude")
            lon_elem = t.find("longitude")

            if (
                name_elem is not None
                and name_elem.text
                and lat_elem is not None
                and lat_elem.text
                and lon_elem is not None
                and lon_elem.text
            ):
                toponym_name = name_elem.text.strip()
                try:
                    gold_lat = float(lat_elem.text.strip())
                    gold_lon = float(lon_elem.text.strip())
                except ValueError:
                    continue

                total_evaluated += 1
                pred_loc = gazetteer.resolve_toponym(toponym_name, context_text=art_text)

                if pred_loc is not None:
                    resolved_count += 1
                    dist = haversine_km(pred_loc.latitude, pred_loc.longitude, gold_lat, gold_lon)
                    error_distances.append(dist)

                    if dist <= distance_threshold_km:
                        accurate_161_count += 1
                    if dist <= 50.0:
                        accurate_50_count += 1

                if max_eval and total_evaluated >= max_eval:
                    break
        if max_eval and total_evaluated >= max_eval:
            break

    resolution_rate = resolved_count / total_evaluated if total_evaluated > 0 else 0.0
    accuracy_161 = accurate_161_count / resolved_count if resolved_count > 0 else 0.0
    accuracy_50 = accurate_50_count / resolved_count if resolved_count > 0 else 0.0

    sorted_dists = sorted(error_distances)
    median_error_km = sorted_dists[len(sorted_dists) // 2] if sorted_dists else float("nan")
    mean_error_km = sum(error_distances) / len(error_distances) if error_distances else float("nan")

    return {
        "benchmark": "GeoWebNews (Gritta et al., 2018)",
        "total_toponyms": total_evaluated,
        "resolved_toponyms": resolved_count,
        "resolution_rate": round(resolution_rate, 4),
        "accuracy_at_161km": round(accuracy_161, 4),
        "accuracy_at_50km": round(accuracy_50, 4),
        "median_error_km": round(median_error_km, 2),
        "mean_error_km": round(mean_error_km, 2),
    }


def regenerate_table_vi(geotagger: Optional[Geotagger] = None) -> List[Dict[str, str]]:
    """Regenerate Table VI (Geo-tagging results from textual and metadata cues).

    Corresponds to Claim C009 in the claims ledger.
    """
    if geotagger is None:
        geotagger = Geotagger()

    # The exact sentences from Table VI of the monograph
    monograph_sentences = [
        "Flooding in New York City caused significant damage.",
        "Earthquake in Tokyo left thousands stranded.",
        "Wildfire spread across the California coastline.",
    ]

    rows = []
    for sentence in monograph_sentences:
        locations = geotagger.resolve_text_locations(sentence)
        if locations:
            # Take primary resolved location
            primary_loc = locations[0]
            resolved_str = primary_loc.to_formatted_str()
        else:
            resolved_str = "Unresolved"

        rows.append({
            "Input Sentence": sentence,
            "Resolved Location": resolved_str,
        })

    return rows
