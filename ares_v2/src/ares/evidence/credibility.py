"""Source credibility weighting operator W(source)."""

from typing import Dict, Optional, Set
from urllib.parse import urlparse

# Fact-checking domains to strictly exclude in web-retrieval evaluation (Red Flag RF2)
FACTCHECK_EXCLUSION_DOMAINS: Set[str] = {
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "afp.com",
    "reuters.com/fact-check",
    "apnews.com/hub/ap-fact-check",
    "boomlive.in",
    "altnews.in",
    "factly.in",
    "vishvasnews.com",
    "checkyourfact.com",
    "leadstories.com",
}

# Empirical domain credibility scores (Lin et al. 2023 / NewsGuard standards)
DEFAULT_CREDIBILITY_RATINGS: Dict[str, float] = {
    "who.int": 0.98,
    "cdc.gov": 0.98,
    "nature.com": 0.98,
    "science.org": 0.98,
    "nejm.org": 0.98,
    "thelancet.com": 0.98,
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "bbc.com": 0.93,
    "bbc.co.uk": 0.93,
    "nytimes.com": 0.90,
    "washingtonpost.com": 0.88,
    "theguardian.com": 0.88,
    "wikipedia.org": 0.85,
    "corpus": 0.85, # Default for frozen Wikipedia / academic corpus
}

class CredibilityScorer:
    """Computes source credibility weight W in [0, 1]."""
    
    def __init__(self, ratings: Optional[Dict[str, float]] = None, default_weight: float = 0.50):
        self.ratings = ratings if ratings is not None else DEFAULT_CREDIBILITY_RATINGS.copy()
        self.default_weight = default_weight
        
    def _extract_domain(self, source: str) -> str:
        s_lower = source.strip().lower()
        if "://" in s_lower:
            parsed = urlparse(s_lower)
            netloc = parsed.netloc
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc
        return s_lower
        
    def is_factcheck_domain(self, source: str) -> bool:
        s_lower = source.strip().lower()
        domain = self._extract_domain(source)
        for fc in FACTCHECK_EXCLUSION_DOMAINS:
            if "/" in fc:
                if fc in s_lower:
                    return True
            else:
                if domain == fc or domain.endswith("." + fc):
                    return True
        return False
        
    def score(self, source: str) -> float:
        domain = self._extract_domain(source)
        if domain in self.ratings:
            return self.ratings[domain]
            
        # Match parent domain suffix
        for key, val in self.ratings.items():
            if domain.endswith("." + key) or domain == key:
                return val
                
        # Default unrated domain weight
        return self.default_weight
