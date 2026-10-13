"""Mathematical Audit of Table XI Audio Rows (Claims C014, C015).

Formal derivation of the closed-form relationship between True Positive Rate (TPR),
False Positive Rate (FPR), Class Imbalance Ratio (r = N_bonafide / N_spoof), and F1 Score.
Resolves Audit Issue I-14 for the audio modalities:
  1. ARES (Voice Embedding): TPR=97%, FPR=3.0%, Reported F1=0.98
  2. Traditional Audio ML: TPR=83%, FPR=12.0%, Reported F1=0.88
"""

from dataclasses import dataclass
import pandas as pd


def calculate_precision(tpr: float, fpr: float, r: float = 1.0) -> float:
    """Calculate precision from TPR, FPR, and bonafide-to-spoof sample ratio r.
    
    Precision = TP / (TP + FP) = (TPR * N_spoof) / (TPR * N_spoof + FPR * N_bonafide)
              = TPR / (TPR + r * FPR)
    """
    denom = tpr + r * fpr
    if denom <= 0:
        return 0.0
    return tpr / denom


def calculate_f1(tpr: float, fpr: float, r: float = 1.0) -> float:
    """Calculate F1 score from TPR, FPR, and bonafide-to-spoof sample ratio r.
    
    F1 = 2 * Precision * Recall / (Precision + Recall)
       = 2 * TPR / (1 + TPR + r * FPR)
    """
    denom = 1.0 + tpr + r * fpr
    if denom <= 0:
        return 0.0
    return (2.0 * tpr) / denom


def solve_class_ratio_for_f1(tpr: float, fpr: float, target_f1: float) -> float:
    """Solve for the bonafide-to-spoof ratio r that yields target_f1 given TPR and FPR.
    
    target_f1 = 2 * TPR / (1 + TPR + r * FPR)
    => 1 + TPR + r * FPR = 2 * TPR / target_f1
    => r * FPR = 2 * TPR / target_f1 - (1 + TPR)
    => r = [2 * TPR / target_f1 - (1 + TPR)] / FPR
    """
    if fpr <= 0 or target_f1 <= 0:
        raise ValueError("FPR and target_f1 must be positive")
    numerator = (2.0 * tpr / target_f1) - (1.0 + tpr)
    return numerator / fpr


@dataclass(frozen=True)
class Table11AudioAuditResult:
    model_name: str
    reported_tpr: float
    reported_fpr: float
    reported_f1: float
    balanced_precision: float
    balanced_f1: float
    balanced_discrepancy: float
    solved_ratio_for_reported_f1: float
    implied_spoof_prevalence_pct: float


def audit_table11_audio_rows() -> pd.DataFrame:
    """Audit the audio rows of Table XI against mathematical ground truth."""
    cases = [
        ("ARES (Voice Embedding)", 0.97, 0.030, 0.98),
        ("Traditional Audio ML", 0.83, 0.120, 0.88),
    ]
    
    results = []
    for model, tpr, fpr, rep_f1 in cases:
        prec_bal = calculate_precision(tpr, fpr, r=1.0)
        f1_bal = calculate_f1(tpr, fpr, r=1.0)
        disc_bal = rep_f1 - f1_bal
        r_solved = solve_class_ratio_for_f1(tpr, fpr, rep_f1)
        pi_solved = 1.0 / (1.0 + r_solved) * 100.0
        
        results.append({
            "model": model,
            "reported_tpr": tpr,
            "reported_fpr": fpr,
            "reported_f1": rep_f1,
            "balanced_precision_r1": prec_bal,
            "balanced_implied_f1_r1": f1_bal,
            "balanced_discrepancy": disc_bal,
            "implied_ratio_r": r_solved,
            "implied_spoof_prevalence_pct": pi_solved,
        })
        
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = audit_table11_audio_rows()
    print(df.to_string(index=False))
