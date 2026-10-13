"""Evidence functional Truth'(C) = Sum(S * N * W) / Sum(S) and verdict decisions."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class EvidenceItemTrace:
    evidence_id: str
    text: str
    source: str
    similarity_score: float   # S_i in [0, 1]
    nli_stance_score: float   # N_i in [-1, +1]
    credibility_weight: float # W_i in [0, 1]
    effective_weight: float   # S_i * W_i
    contribution: float       # S_i * N_i * W_i
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EvidencePipelineOutput:
    claim: str
    truth_score: float             # Truth' in [-1, +1]
    legacy_confidence: float       # |Truth'| * log(1 + K)
    calibrated_confidence: float   # Sigmoid-calibrated posterior P(SUPPORTS)
    verdict: str                   # "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"
    evidence_count: int
    traces: List[EvidenceItemTrace]

class EvidenceFunctional:
    """Computes the ARES Evidence Functional and evaluates claim verdicts."""
    
    def __init__(
        self,
        tau_true: float = 0.20,
        tau_false: float = -0.20,
        min_evidence_threshold: float = 0.10,
        calibration_temp: float = 2.5
    ):
        self.tau_true = tau_true
        self.tau_false = tau_false
        self.min_evidence_threshold = min_evidence_threshold
        self.calibration_temp = calibration_temp
        
    def compute(
        self,
        claim: str,
        evidence_items: List[Dict[str, Any]],
        use_soft_n: bool = True
    ) -> EvidencePipelineOutput:
        """
        evidence_items must each contain:
          - 'id' (or 'doc_id')
          - 'text'
          - 'source'
          - 'similarity' (S_i)
          - 'nli' (dict with 'soft_n' and 'hard_n')
          - 'credibility' (W_i)
        """
        if not evidence_items:
            return EvidencePipelineOutput(
                claim=claim,
                truth_score=0.0,
                legacy_confidence=0.0,
                calibrated_confidence=0.50,
                verdict="NOT_ENOUGH_INFO",
                evidence_count=0,
                traces=[]
            )
            
        traces: List[EvidenceItemTrace] = []
        weighted_stance_sum = 0.0
        similarity_sum = 0.0
        
        for item in evidence_items:
            eid = str(item.get("id", item.get("doc_id", "0")))
            text = str(item.get("text", ""))
            source = str(item.get("source", "corpus"))
            
            s_i = float(item.get("similarity", 1.0))
            w_i = float(item.get("credibility", 1.0))
            
            nli_dict = item.get("nli", {})
            if isinstance(nli_dict, dict):
                n_i = float(nli_dict.get("soft_n" if use_soft_n else "hard_n", 0.0))
            else:
                n_i = float(nli_dict)
                
            eff_weight = s_i * w_i
            contrib = eff_weight * n_i
            
            weighted_stance_sum += contrib
            similarity_sum += s_i
            
            traces.append(
                EvidenceItemTrace(
                    evidence_id=eid,
                    text=text,
                    source=source,
                    similarity_score=s_i,
                    nli_stance_score=n_i,
                    credibility_weight=w_i,
                    effective_weight=eff_weight,
                    contribution=contrib,
                    metadata=item.get("metadata")
                )
            )
            
        k = len(traces)
        if similarity_sum > 0:
            truth_score = float(np.clip(weighted_stance_sum / similarity_sum, -1.0, 1.0))
        else:
            truth_score = 0.0
            
        # Legacy uncalibrated confidence (can exceed 1.0 for large K, documented in C078)
        legacy_conf = float(abs(truth_score) * np.log(1.0 + k))
        
        # Calibrated posterior confidence via logistic scaling
        calibrated_conf = float(1.0 / (1.0 + np.exp(-self.calibration_temp * truth_score)))
        
        # Verdict decision rule
        if truth_score >= self.tau_true and similarity_sum >= self.min_evidence_threshold:
            verdict = "SUPPORTS"
        elif truth_score <= self.tau_false and similarity_sum >= self.min_evidence_threshold:
            verdict = "REFUTES"
        else:
            verdict = "NOT_ENOUGH_INFO"
            
        return EvidencePipelineOutput(
            claim=claim,
            truth_score=truth_score,
            legacy_confidence=legacy_conf,
            calibrated_confidence=calibrated_conf,
            verdict=verdict,
            evidence_count=k,
            traces=traces
        )
