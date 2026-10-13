"""Symbolic quantity, date, and numerical extraction and alignment."""

import re
from typing import List, Dict, Any, Optional, Tuple
import dateparser

class QuantityExtractor:
    """Extracts numbers, percentages, currencies, and dates from text."""
    
    NUM_REGEX = re.compile(r'\b(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\b')
    PERCENT_REGEX = re.compile(r'\b(?:\d+(?:\.\d+)?)\s*%\b')
    YEAR_REGEX = re.compile(r'\b(?:19|20)\d{2}\b')
    
    def extract_numbers(self, text: str) -> List[float]:
        matches = self.NUM_REGEX.findall(text)
        numbers = []
        for m in matches:
            clean = m.replace(',', '')
            try:
                numbers.append(float(clean))
            except ValueError:
                continue
        return numbers
        
    def extract_years(self, text: str) -> List[int]:
        matches = self.YEAR_REGEX.findall(text)
        return [int(m) for m in matches]
        
    def compare_quantities(
        self,
        claim_text: str,
        evidence_text: str,
        rel_tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Check for numerical contradictions between claim and evidence.
        Returns contradiction count and details.
        """
        claim_nums = self.extract_numbers(claim_text)
        ev_nums = self.extract_numbers(evidence_text)
        
        contradictions = []
        confirmations = []
        
        # If claim asserts specific numbers not present or contradicting evidence
        for c_num in claim_nums:
            matched = False
            for e_num in ev_nums:
                diff = abs(c_num - e_num)
                denom = max(abs(c_num), abs(e_num), 1.0)
                if diff / denom <= rel_tolerance:
                    matched = True
                    confirmations.append((c_num, e_num))
                    break
            if not matched and len(ev_nums) > 0:
                contradictions.append(c_num)
                
        return {
            "claim_numbers": claim_nums,
            "evidence_numbers": ev_nums,
            "confirmations": confirmations,
            "potential_contradictions": contradictions,
            "has_numeric_conflict": len(contradictions) > 0 and len(confirmations) == 0
        }
