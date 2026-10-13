"""Video-level Frame Score Aggregation, Operating Thresholds & Metrics.

Operationalizes:
- Mean frame probability pooling (Section X / Appendix A declared operator).
- Top-k entropy frame aggregation (Appendix D variant).
- Validation-only threshold selection (Youden's J, FPR=1%, FPR=5%).
- Video-level ROC-AUC, PR-AUC, Accuracy, TPR, FPR, and F1 calculation with percentile bootstrap CIs.
"""

from typing import Optional, Union
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def mean_pooling(frame_scores: Union[list[float], np.ndarray]) -> float:
    """Compute video-level score as the mean probability over sampled frames."""
    scores = np.asarray(frame_scores, dtype=np.float64)
    if scores.size == 0:
        return 0.5  # Neutral fallback for empty frame sequence
    return float(np.mean(scores))


def topk_entropy_pooling(frame_scores: Union[list[float], np.ndarray], k: int = 5) -> float:
    """Compute video-level score using top-k entropy frames (Appendix D variant).
    
    Frames with high entropy H(p) = -p*log(p) - (1-p)*log(1-p) represent the most
    uncertain / ambiguous evidence; pooling their predictions focuses on boundary decisions.
    """
    scores = np.asarray(frame_scores, dtype=np.float64)
    if scores.size == 0:
        return 0.5
    if scores.size <= k:
        return float(np.mean(scores))
    
    eps = 1e-7
    clamped = np.clip(scores, eps, 1.0 - eps)
    entropy = -(clamped * np.log(clamped) + (1.0 - clamped) * np.log(1.0 - clamped))
    top_indices = np.argsort(entropy)[-k:]
    return float(np.mean(scores[top_indices]))


def find_operating_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    criterion: str = "youden",
) -> tuple[float, dict[str, float]]:
    """Select decision threshold on validation set according to pre-registered criterion.
    
    Criteria:
    - 'youden': Maximizes Youden's J = TPR - FPR.
    - 'fixed_fpr_5': Threshold where FPR <= 0.05.
    - 'fixed_fpr_1': Threshold where FPR <= 0.01.
    """
    fprs, tprs, thresholds = roc_curve(y_true, y_score)
    
    if criterion == "youden":
        j_scores = tprs - fprs
        best_idx = int(np.argmax(j_scores))
        chosen_thresh = float(thresholds[best_idx])
    elif criterion == "fixed_fpr_5":
        valid_indices = np.where(fprs <= 0.05)[0]
        if len(valid_indices) == 0:
            best_idx = 0
        else:
            best_idx = int(valid_indices[-1])
        chosen_thresh = float(thresholds[best_idx])
    elif criterion == "fixed_fpr_1":
        valid_indices = np.where(fprs <= 0.01)[0]
        if len(valid_indices) == 0:
            best_idx = 0
        else:
            best_idx = int(valid_indices[-1])
        chosen_thresh = float(thresholds[best_idx])
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    y_pred = (y_score >= chosen_thresh).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    precision = tp / max(1, tp + fp)
    recall = tpr
    f1 = (2 * precision * recall) / max(1e-7, precision + recall)
    
    stats = {
        "threshold": chosen_thresh,
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    return chosen_thresh, stats


def compute_video_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    r: Optional[float] = None,
) -> dict[str, float]:
    """Compute comprehensive video-level evaluation metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    
    roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5
    
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    
    tpr = float(tp / max(1, tp + fn))
    fpr = float(fp / max(1, fp + tn))
    accuracy = float((tp + tn) / max(1, len(y_true)))
    precision = float(tp / max(1, tp + fp))
    recall = tpr
    f1 = float((2 * precision * recall) / max(1e-7, precision + recall))
    
    # Balanced F1 (r = 1.0)
    balanced_f1 = float((2.0 * tpr) / max(1e-7, 1.0 + tpr + 1.0 * fpr))
    
    # Operating TPR at FPR = 1% and 5%
    fprs, tprs, _ = roc_curve(y_true, y_score)
    tpr_at_fpr_1 = float(np.interp(0.01, fprs, tprs))
    tpr_at_fpr_5 = float(np.interp(0.05, fprs, tprs))
    
    out = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_f1": balanced_f1,
        "tpr_at_fpr_1": tpr_at_fpr_1,
        "tpr_at_fpr_5": tpr_at_fpr_5,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_samples": len(y_true),
    }
    if r is not None:
        out["stated_ratio_f1"] = float((2.0 * tpr) / max(1e-7, 1.0 + tpr + r * fpr))
    return out


def bootstrap_video_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute 95% percentile bootstrap confidence intervals over videos."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    
    boot_aucs = []
    boot_f1s = []
    boot_accs = []
    boot_tprs = []
    boot_fprs = []
    
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_true = y_true[idx]
        sample_score = y_score[idx]
        
        if len(np.unique(sample_true)) < 2:
            continue
            
        m = compute_video_metrics(sample_true, sample_score, threshold=threshold)
        boot_aucs.append(m["roc_auc"])
        boot_f1s.append(m["f1"])
        boot_accs.append(m["accuracy"])
        boot_tprs.append(m["tpr"])
        boot_fprs.append(m["fpr"])
        
    return {
        "roc_auc": (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))),
        "f1": (float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5))),
        "accuracy": (float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5))),
        "tpr": (float(np.percentile(boot_tprs, 2.5)), float(np.percentile(boot_tprs, 97.5))),
        "fpr": (float(np.percentile(boot_fprs, 2.5)), float(np.percentile(boot_fprs, 97.5))),
    }
