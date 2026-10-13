"""Semantic similarity and passage ranking operator S(C, E_i)."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class BaseRanker(ABC):
    """Abstract base class for semantic relevance scoring."""
    
    @abstractmethod
    def score(self, claim: str, passages: List[str]) -> List[float]:
        """Return relevance scores S in [0, 1] for each passage given the claim."""
        pass

class BiEncoderRanker(BaseRanker):
    """Bi-encoder cosine similarity ranker."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        
    def score(self, claim: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
            
        claim_emb = self.model.encode([claim], normalize_embeddings=True)[0]
        passage_embs = self.model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        
        # Cosine similarity is dot product of normalized vectors
        cos_sims = np.dot(passage_embs, claim_emb)
        # Normalize from [-1, 1] to [0, 1]
        scores = (cos_sims + 1.0) / 2.0
        return [float(np.clip(s, 0.0, 1.0)) for s in scores]

class CrossEncoderRanker(BaseRanker):
    """Cross-encoder relevance ranker."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)
        
    def score(self, claim: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
            
        pairs = [(claim, p) for p in passages]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Apply sigmoid to map real-valued logits to [0, 1]
        probs = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))
        return [float(p) for p in probs]
