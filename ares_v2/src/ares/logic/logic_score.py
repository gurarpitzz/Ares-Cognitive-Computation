"""Symbolic logic scoring and contradiction aggregation."""

from typing import Dict, Any, List
from .quantities import QuantityExtractor

class SymbolicLogicLayer:
    """Combines rule-based consistency, numerical checks, and entity constraints."""
    
    def __init__(self):
        self.quantity_extractor = QuantityExtractor()
        
    def evaluate(self, claim: str, evidence_passages: List[str]) -> Dict[str, Any]:
        """
        Evaluate symbolic consistency across evidence passages.
        """
        total_numeric_conflicts = 0
        total_confirmations = 0
        
        for passage in evidence_passages:
            cmp_res = self.quantity_extractor.compare_quantities(claim, passage)
            if cmp_res["has_numeric_conflict"]:
                total_numeric_conflicts += 1
            if len(cmp_res["confirmations"]) > 0:
                total_confirmations += 1
                
        # Determine symbolic adjustment
        if total_numeric_conflicts > 0 and total_confirmations == 0:
            logic_flag = "CONTRADICTION"
            logic_penalty = -0.30
        elif total_confirmations > 0 and total_numeric_conflicts == 0:
            logic_flag = "CONSISTENT"
            logic_penalty = +0.10
        else:
            logic_flag = "NEUTRAL"
            logic_penalty = 0.0
            
        return {
            "logic_flag": logic_flag,
            "logic_adjustment": logic_penalty,
            "numeric_conflicts": total_numeric_conflicts,
            "numeric_confirmations": total_confirmations
        }
