"""Mathematical Audit of Table XI (Visual Deepfake Claims C013, C015).

Formal derivation of the closed-form relationship between True Positive Rate (TPR),
False Positive Rate (FPR), Class Imbalance Ratio (r = N_real / N_fake), and F1 Score.
Resolves Audit Issue I-14.
"""

from dataclasses import dataclass
import pandas as pd


def calculate_precision(tpr: float, fpr: float, r: float = 1.0) -> float:
    """Calculate precision from TPR, FPR, and real-to-fake sample ratio r.
    
    Precision = TP / (TP + FP) = (TPR * N_fake) / (TPR * N_fake + FPR * N_real)
              = TPR / (TPR + r * FPR)
    """
    denom = tpr + r * fpr
    if denom <= 0:
        return 0.0
    return tpr / denom


def calculate_f1(tpr: float, fpr: float, r: float = 1.0) -> float:
    """Calculate F1 score from TPR, FPR, and real-to-fake sample ratio r.
    
    F1 = 2 * Precision * Recall / (Precision + Recall)
       = 2 * TPR / (1 + TPR + r * FPR)
    """
    denom = 1.0 + tpr + r * fpr
    if denom <= 0:
        return 0.0
    return (2.0 * tpr) / denom


def solve_class_ratio_for_f1(tpr: float, fpr: float, target_f1: float) -> float:
    """Solve for the real-to-fake ratio r that yields target_f1 given TPR and FPR.
    
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
class Table11AuditResult:
    model_name: str
    reported_tpr: float
    reported_fpr: float
    reported_f1: float
    balanced_precision: float
    balanced_f1: float
    balanced_discrepancy: float
    ffpp_test_f1: float
    solved_ratio_for_reported_f1: float
    solved_fake_prior: float


def run_table11_audit() -> list[Table11AuditResult]:
    """Execute complete formal audit of Table XI claims."""
    # ARES Deepfake row (C013)
    ares_tpr = 0.95  # ledger:C013 hyperparameter
    ares_fpr = 0.025  # ledger:C013 hyperparameter
    ares_f1 = 0.97  # ledger:C013 hyperparameter
    
    # CNN Baseline row (C015)
    cnn_tpr = 0.89  # ledger:C015 hyperparameter
    cnn_fpr = 0.08  # ledger:C015 hyperparameter
    cnn_f1 = 0.92  # ledger:C015 hyperparameter
    
    ffpp_r = 140.0 / 560.0  # Official FaceForensics++ c23 test split real:fake ratio = 0.25
    
    records = []
    for name, tpr, fpr, f1 in [
        ("ARES (ResNet-50)", ares_tpr, ares_fpr, ares_f1),
        ("CNN Baseline (Xception)", cnn_tpr, cnn_fpr, cnn_f1),
    ]:
        prec_bal = calculate_precision(tpr, fpr, r=1.0)
        f1_bal = calculate_f1(tpr, fpr, r=1.0)
        f1_ffpp = calculate_f1(tpr, fpr, r=ffpp_r)
        solved_r = solve_class_ratio_for_f1(tpr, fpr, f1)
        solved_prior = 1.0 / (1.0 + solved_r)
        
        records.append(
            Table11AuditResult(
                model_name=name,
                reported_tpr=tpr,
                reported_fpr=fpr,
                reported_f1=f1,
                balanced_precision=prec_bal,
                balanced_f1=f1_bal,
                balanced_discrepancy=f1 - f1_bal,
                ffpp_test_f1=f1_ffpp,
                solved_ratio_for_reported_f1=solved_r,
                solved_fake_prior=solved_prior,
            )
        )
    return records


def generate_audit_table_df() -> pd.DataFrame:
    """Generate pandas DataFrame summarizing the audit results."""
    records = run_table11_audit()
    data = []
    for r in records:
        data.append({
            "model": r.model_name,
            "reported_tpr": f"{r.reported_tpr*100:.1f}%",
            "reported_fpr": f"{r.reported_fpr*100:.1f}%",
            "reported_f1": f"{r.reported_f1:.4f}",
            "balanced_precision": f"{r.balanced_precision:.4f}",
            "balanced_implied_f1": f"{r.balanced_f1:.4f}",
            "balanced_error": f"{r.balanced_discrepancy:+.4f}",
            "ffpp_test_ratio_f1": f"{r.ffpp_test_f1:.4f}",
            "implied_real_fake_ratio": f"{r.solved_ratio_for_reported_f1:.4f}",
            "implied_fake_prior": f"{r.solved_fake_prior*100:.2f}%",
        })
    return pd.DataFrame(data)
