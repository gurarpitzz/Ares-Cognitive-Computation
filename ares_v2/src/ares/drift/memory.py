"""Temporal Claim Memory Module with Strict Non-Leakage Enforcement (RF1, M06).

Maintains a chronological memory of fact-checked claims and matches incoming claims
strictly against past evidence (t_memory < t_query).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class TemporalClaimMemory:
    """Chronologically ordered fact-check claim memory store."""
    
    def __init__(self, theta_match: float = 0.85):
        self.theta_match = theta_match
        self.item_ids: List[str] = []
        self.claims: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.verdicts: List[float] = []  # +1 for SUPPORTS/TRUE, -1 for REFUTES/FALSE
        self.timestamps: List[int] = []  # integer timestamps (e.g. days or unix seconds)
        
    def add_claim(
        self,
        item_id: str,
        claim_text: str,
        embedding: np.ndarray,
        verdict: float,
        timestamp: int
    ):
        """Insert a fact-checked claim into historical memory."""
        # Normalize embedding to unit length
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-12)
        
        self.item_ids.append(item_id)
        self.claims.append(claim_text)
        self.embeddings.append(norm_emb)
        self.verdicts.append(float(verdict))
        self.timestamps.append(int(timestamp))
        
    def query(
        self,
        query_embedding: np.ndarray,
        query_timestamp: int,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Query memory for past fact-checked claims strictly prior to query_timestamp.
        
        Enforces Red Flag RF1: t_memory < t_query.
        """
        if len(self.timestamps) == 0:
            return self._empty_result()
            
        timestamps_arr = np.array(self.timestamps)
        # Strict temporal non-leakage constraint
        valid_indices = np.where(timestamps_arr < query_timestamp)[0]
        
        if len(valid_indices) == 0:
            return self._empty_result()
            
        # Compute cosine similarities against strictly past memories
        norm_q = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
        past_embs = np.array([self.embeddings[i] for i in valid_indices])
        sims = past_embs @ norm_q
        
        # Sort descending
        top_sub_idx = np.argsort(sims)[::-1][:top_k]
        top_valid_idx = valid_indices[top_sub_idx]
        top_sims = sims[top_sub_idx]
        
        max_sim = float(top_sims[0]) if len(top_sims) > 0 else 0.0
        n_matches = int(np.sum(top_sims >= self.theta_match))
        
        # Weighted verdict score: Sum(S_i * V_i) / Sum(S_i)
        weights = np.maximum(0.0, top_sims)
        total_w = np.sum(weights)
        if total_w > 0:
            top_verdicts = np.array([self.verdicts[i] for i in top_valid_idx])
            weighted_score = float(np.sum(weights * top_verdicts) / total_w)
        else:
            weighted_score = 0.0
            
        matched_records = []
        for rank, (idx, sim) in enumerate(zip(top_valid_idx, top_sims)):
            matched_records.append({
                "rank": rank + 1,
                "item_id": self.item_ids[idx],
                "claim": self.claims[idx],
                "similarity": float(sim),
                "verdict": self.verdicts[idx],
                "timestamp": self.timestamps[idx],
            })
            
        return {
            "has_history": True,
            "max_similarity": max_sim,
            "n_matches_above_theta": n_matches,
            "memory_veracity_score": weighted_score,
            "top_k_matches": matched_records,
        }
        
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "has_history": False,
            "max_similarity": 0.0,
            "n_matches_above_theta": 0,
            "memory_veracity_score": 0.0,
            "top_k_matches": [],
        }


def evaluate_claim_memory_map(
    memory: TemporalClaimMemory,
    queries: List[Dict[str, Any]],
    top_k: int = 5
) -> Dict[str, float]:
    """Compute Mean Average Precision @ k (MAP@k) for claim retrieval."""
    aps = []
    recalls = []
    
    for q in queries:
        q_emb = q["embedding"]
        q_time = q["timestamp"]
        gold_id = q.get("gold_match_id")
        
        res = memory.query(q_emb, q_time, top_k=top_k)
        if not res["has_history"]:
            aps.append(0.0)
            recalls.append(0.0)
            continue
            
        matches = [m["item_id"] for m in res["top_k_matches"]]
        if gold_id in matches:
            rank = matches.index(gold_id) + 1
            aps.append(1.0 / rank)
            recalls.append(1.0)
        else:
            aps.append(0.0)
            recalls.append(0.0)
            
    return {
        f"map@{top_k}": float(np.mean(aps)) if aps else 0.0,
        f"recall@{top_k}": float(np.mean(recalls)) if recalls else 0.0,
        "n_queries": len(queries),
    }
