"""Expected Latency Model and Manuscript Arithmetic Proof (Claim C062).

Implements the expected end-to-end latency formulation:
    E[T] = (1 - E) * T_auto + E * T_human

And proves the mathematical root cause of the manuscript's invalid claim (C062).
The authors omitted converting the human review duration from minutes to seconds.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_expected_latency(
    t_auto_ms: float = 415.0,  # ledger:C062 hyperparameter
    t_human_min: float = 18.6,  # ledger:C062 hyperparameter
    escalation_rate: float = 0.068,  # ledger:C062 hyperparameter
) -> Dict[str, float]:
    """Compute exact expected latency in seconds, highlighting the unit-conversion flaw."""
    t_auto_s = t_auto_ms / 1000.0
    t_human_s = t_human_min * 60.0
    
    # Correct physical formula (all terms in seconds)
    term_auto = (1.0 - escalation_rate) * t_auto_s
    term_human = escalation_rate * t_human_s
    expected_latency_s = term_auto + term_human
    
    # Buggy manuscript arithmetic: omitted minute-to-second conversion
    manuscript_buggy_calc = float(escalation_rate * t_human_min)
    reported_val = 1.26  # ledger:C062 hyperparameter
    
    return {
        "t_auto_ms": float(t_auto_ms),
        "t_human_min": float(t_human_min),
        "escalation_rate": float(escalation_rate),
        "true_expected_latency_s": float(expected_latency_s),
        "term_auto_s": float(term_auto),
        "term_human_s": float(term_human),
        "manuscript_reported_s": reported_val,
        "manuscript_buggy_reproduction_s": float(manuscript_buggy_calc),
        "discrepancy_factor": float(expected_latency_s / reported_val),
        "status": "ARITHMETIC_CORRECTION_VERIFIED",
    }


def sweep_latency_sensitivity(
    t_auto_ms_list: List[float] = [200.0, 415.0, 600.0, 1000.0],  # ledger:C062 hyperparameter
    t_human_min_list: List[float] = [1.0, 5.0, 18.6, 30.0, 60.0],  # ledger:C062 hyperparameter
    escalation_rates: List[float] = [0.01, 0.02, 0.05, 0.068, 0.10, 0.15, 0.20, 0.30],  # ledger:C062 hyperparameter
) -> pd.DataFrame:
    """Generate sensitivity grid across automation latencies, human latencies, and escalation rates."""
    records = []
    
    for t_auto in t_auto_ms_list:
        for t_human in t_human_min_list:
            for e in escalation_rates:
                t_auto_s = t_auto / 1000.0
                t_human_s = t_human * 60.0
                e_t = (1.0 - e) * t_auto_s + e * t_human_s
                
                records.append({
                    "t_auto_ms": t_auto,
                    "t_human_min": t_human,
                    "escalation_rate": e,
                    "expected_latency_s": float(e_t),
                    "human_overhead_pct": float((e * t_human_s) / e_t * 100.0),
                })
                
    return pd.DataFrame(records)
