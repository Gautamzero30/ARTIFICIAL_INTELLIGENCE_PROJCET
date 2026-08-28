"""
Evaluator orchestrator for running offline and online benchmark evaluations.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from src.core.logging import get_logger
from src.evaluation.metrics import (
    EvaluationMetrics,
    compute_binary_metrics,
    compute_roc_and_pr_curves,
)
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_distribution,
)

logger = get_logger("authentica.evaluator")


class DatasetEvaluator:
    """
    Evaluates detector performance over labeled test sets and persists
    calculated metrics and publication figures.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.reports_dir = output_dir or (base_dir / "reports")
        self.metrics_dir = self.reports_dir / "metrics"
        self.figures_dir = self.reports_dir / "figures"

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_predictions(
        self,
        modality: str,
        model_name: str,
        y_true: List[int],
        y_scores: List[float],
        threshold: float = 0.50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates metrics from real predictions, renders figures, and saves summary JSON.
        """
        logger.info(f"Computing evaluation metrics for modality '{modality}' ({len(y_true)} samples)...")

        metrics: EvaluationMetrics = compute_binary_metrics(
            y_true=y_true,
            y_scores=y_scores,
            threshold=threshold,
        )

        curves = compute_roc_and_pr_curves(y_true=y_true, y_scores=y_scores)

        # Generate figures
        cm_path = self.figures_dir / f"{modality}_confusion_matrix.png"
        plot_confusion_matrix(
            confusion_matrix_data=metrics.confusion_matrix,
            output_path=cm_path,
            title=f"{modality.capitalize()} Detector — Confusion Matrix",
        )

        if "roc_curve" in curves:
            roc_path = self.figures_dir / f"{modality}_roc_curve.png"
            plot_roc_curve(
                fpr=curves["roc_curve"]["fpr"],
                tpr=curves["roc_curve"]["tpr"],
                auc_score=metrics.roc_auc,
                output_path=roc_path,
                title=f"{modality.capitalize()} Detector — ROC Curve",
            )

        dist_path = self.figures_dir / f"{modality}_score_distribution.png"
        plot_score_distribution(
            y_true=y_true,
            y_scores=y_scores,
            output_path=dist_path,
            title=f"{modality.capitalize()} AI Score Distribution",
        )

        report_payload = {
            "modality": modality,
            "model_name": model_name,
            "metrics": metrics.to_dict(),
            "curves": curves,
            "metadata": metadata or {},
            "figures": {
                "confusion_matrix": str(cm_path),
                "roc_curve": str(self.figures_dir / f"{modality}_roc_curve.png") if "roc_curve" in curves else None,
                "score_distribution": str(dist_path),
            },
        }

        # Save metrics to reports/metrics/<modality>_metrics.json
        json_path = self.metrics_dir / f"{modality}_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

        logger.info(f"Evaluation report saved to {json_path}")
        return report_payload
