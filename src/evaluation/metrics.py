"""
Mathematical metrics calculations for classification evaluation.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class EvaluationMetrics:
    """
    Standard evaluation metrics summary for binary detection tasks.
    """
    sample_count: int
    positive_count: int  # AI samples
    negative_count: int  # Human samples
    threshold: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    roc_auc: Optional[float]
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "threshold": round(self.threshold, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "f1_score": round(self.f1_score, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "confusion_matrix": self.confusion_matrix,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }


def compute_binary_metrics(
    y_true: List[int],
    y_scores: List[float],
    threshold: float = 0.50,
) -> EvaluationMetrics:
    """
    Computes all standard binary classification metrics from true labels and raw continuous scores.
    
    Args:
        y_true: Ground truth binary labels (0 = Human/Real, 1 = AI/Synthetic).
        y_scores: Continuous AI detection scores in [0.0, 1.0].
        threshold: Decision threshold for binarization (default 0.50).
    """
    if len(y_true) != len(y_scores):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_scores ({len(y_scores)})")

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty dataset.")

    y_true_np = np.array(y_true, dtype=int)
    y_scores_np = np.array(y_scores, dtype=float)

    # Binarize predictions based on threshold
    y_pred = (y_scores_np >= threshold).astype(int)

    # Compute base sklearn metrics with zero_division=0 to prevent NaN
    acc = float(accuracy_score(y_true_np, y_pred))
    prec = float(precision_score(y_true_np, y_pred, zero_division=0))
    rec = float(recall_score(y_true_np, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_np, y_pred, zero_division=0))

    # Confusion Matrix [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true_np, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC (requires both classes present in y_true)
    unique_classes = np.unique(y_true_np)
    if len(unique_classes) == 2:
        try:
            auc = float(roc_auc_score(y_true_np, y_scores_np))
        except Exception:
            auc = None
    else:
        auc = None

    return EvaluationMetrics(
        sample_count=len(y_true),
        positive_count=int(np.sum(y_true_np == 1)),
        negative_count=int(np.sum(y_true_np == 0)),
        threshold=threshold,
        accuracy=acc,
        precision=prec,
        recall=rec,
        specificity=specificity,
        f1_score=f1,
        roc_auc=auc,
        confusion_matrix=[[tn, fp], [fn, tp]],
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def compute_roc_and_pr_curves(
    y_true: List[int],
    y_scores: List[float],
) -> Dict[str, Any]:
    """
    Computes coordinates for ROC and Precision-Recall curves.
    """
    y_true_np = np.array(y_true, dtype=int)
    y_scores_np = np.array(y_scores, dtype=float)

    result = {}
    if len(np.unique(y_true_np)) == 2:
        fpr, tpr, roc_thresh = roc_curve(y_true_np, y_scores_np)
        prec, rec, pr_thresh = precision_recall_curve(y_true_np, y_scores_np)

        result["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresh.tolist(),
        }
        result["pr_curve"] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": pr_thresh.tolist(),
        }
    return result
