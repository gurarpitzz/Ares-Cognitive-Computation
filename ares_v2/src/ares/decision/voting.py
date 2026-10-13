"""Heuristic weighted voting engine from manuscript Section VI Algorithm 1."""

from typing import List, Dict, Any, Tuple
import numpy as np

class HeuristicWeightedVoting:
    """Implements Algorithm 1: V(y) = Sum_i delta(y_i = y) * p_i and escalation rule."""
    
    def __init__(self, tau: float = 0.65):
        self.tau = tau
        
    def vote(self, predictions: List[Tuple[int, float]]) -> Dict[str, Any]:
        """
        predictions: list of (predicted_class_int, confidence_float) for each model.
        Returns:
          winner_class: int
          max_vote_score: float
          escalate: bool
          vote_distribution: dict
        """
        votes: Dict[int, float] = {}
        for y_i, p_i in predictions:
            votes[y_i] = votes.get(y_i, 0.0) + p_i
            
        if not votes:
            return {"winner": 0, "confidence": 0.0, "escalate": True, "votes": {}}
            
        # Total weight normalization
        total_p = sum(p for _, p in predictions)
        norm_votes = {cls: score / (total_p + 1e-8) for cls, score in votes.items()}
        
        winner = max(norm_votes.items(), key=lambda x: x[1])
        winner_cls = winner[0]
        max_score = winner[1]
        
        escalate = max_score < self.tau
        
        return {
            "winner": winner_cls,
            "confidence": float(max_score),
            "escalate": escalate,
            "votes": norm_votes
        }
        
    def vote_batch(self, batch_predictions: List[List[Tuple[int, float]]]) -> List[Dict[str, Any]]:
        return [self.vote(sample_preds) for sample_preds in batch_predictions]
