"""Symbolic Wikidata relation checking with temporal qualification."""

from typing import Dict, Any, List, Optional
import re

# Curated factual relations for frozen verification (Wikidata properties)
FACTUAL_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "france": {"capital": "paris", "inception_year": 843, "continent": "europe"},
    "germany": {"capital": "berlin", "inception_year": 1871, "continent": "europe"},
    "india": {"capital": "new delhi", "inception_year": 1947, "continent": "asia"},
    "united states": {"capital": "washington", "inception_year": 1776, "continent": "north america"},
    "united kingdom": {"capital": "london", "inception_year": 1707, "continent": "europe"},
    "japan": {"capital": "tokyo", "inception_year": 660, "continent": "asia"},
    "italy": {"capital": "rome", "inception_year": 1861, "continent": "europe"},
    "canada": {"capital": "ottawa", "inception_year": 1867, "continent": "north america"},
    "jammu and kashmir": {"country": "india", "region": "south asia"},
}

class WikidataRelationChecker:
    """Verifies relational and temporal consistency against frozen entity facts."""
    
    CAPITAL_PATTERN = re.compile(r'capital of ([a-zA-Z\s]+) is ([a-zA-Z\s]+)', re.IGNORECASE)
    LOCATED_PATTERN = re.compile(r'([a-zA-Z\s]+) is located in ([a-zA-Z\s]+)', re.IGNORECASE)
    
    def check_claim(self, claim_text: str) -> Dict[str, Any]:
        c_lower = claim_text.lower()
        
        # Check capital relations (P36)
        cap_match = self.CAPITAL_PATTERN.search(c_lower)
        if cap_match:
            country = cap_match.group(1).strip()
            claimed_capital = cap_match.group(2).strip()
            
            if country in FACTUAL_KNOWLEDGE_BASE:
                gold_capital = FACTUAL_KNOWLEDGE_BASE[country].get("capital")
                if gold_capital:
                    if claimed_capital.lower() in gold_capital.lower() or gold_capital.lower() in claimed_capital.lower():
                        return {"relation": "P36 (capital)", "status": "CONSISTENT", "entity": country, "value": gold_capital}
                    else:
                        return {"relation": "P36 (capital)", "status": "CONTRADICTED", "entity": country, "gold": gold_capital, "claimed": claimed_capital}
                        
        # Check temporal inception relations (P571)
        for country, data in FACTUAL_KNOWLEDGE_BASE.items():
            if country in c_lower and "inception_year" in data:
                inc_year = data["inception_year"]
                # Search for asserted years
                years = [int(y) for y in re.findall(r'\b(?:19|20|18|17)\d{2}\b', claim_text)]
                for yr in years:
                    if "founded in" in c_lower or "established in" in c_lower:
                        if abs(yr - inc_year) > 1:
                            return {"relation": "P571 (inception)", "status": "CONTRADICTED", "entity": country, "gold": inc_year, "claimed": yr}
                            
        return {"relation": None, "status": "UNKNOWN"}
