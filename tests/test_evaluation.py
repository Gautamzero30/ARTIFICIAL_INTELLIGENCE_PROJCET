"""
Unit tests for the evaluation framework and metric calculations.
"""
from pathlib import Path
import pytest

from src.evaluation.evaluator import DatasetEvaluator
from src.evaluation.metrics import (
    EvaluationMetrics,
    compute_binary_metrics,
    compute_roc_and_pr_curves,
)


def test_compute_binary_metrics_perfect_predictions():
    """Verify metrics on a perfect binary classification scenario."""
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_scores = [0.05, 0.10, 0.15, 0.20, 0.85, 0.90, 0.95, 0.99]

    metrics = compute_binary_metrics(y_true, y_scores, threshold=0.50)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.specificity == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.true_positives == 4
    assert metrics.true_negatives == 4
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_compute_binary_metrics_mixed_predictions():
    """Verify metrics on mixed predictions with known FP and FN."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_scores = [0.1, 0.8, 0.2, 0.9, 0.3, 0.7] # 1 FP at idx 1 (0.8), 1 FN at idx 4 (0.3)

    metrics = compute_binary_metrics(y_true, y_scores, threshold=0.50)

    # Preds: [0, 1, 0, 1, 0, 1] -> TN=2, FP=1, FN=1, TP=2
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.accuracy == 4.0 / 6.0
    assert metrics.precision == 2.0 / 3.0
    assert metrics.recall == 2.0 / 3.0


def test_compute_binary_metrics_zero_division_safety():
    """Verify zero division is safely handled when model predicts 0 positives."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.1, 0.1, 0.1] # all below threshold 0.50

    metrics = compute_binary_metrics(y_true, y_scores, threshold=0.50)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


def test_compute_roc_and_pr_curves():
    """Verify ROC and PR curve coordinate extraction."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.2, 0.4, 0.6, 0.8]

    curves = compute_roc_and_pr_curves(y_true, y_scores)
    assert "roc_curve" in curves
    assert "pr_curve" in curves
    assert len(curves["roc_curve"]["fpr"]) > 0
    assert len(curves["roc_curve"]["tpr"]) > 0


def test_dataset_evaluator_persists_files(tmp_path: Path):
    """Verify DatasetEvaluator writes metrics JSON and figures to specified directory."""
    evaluator = DatasetEvaluator(output_dir=tmp_path)

    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]

    report = evaluator.evaluate_predictions(
        modality="image",
        model_name="test-model",
        y_true=y_true,
        y_scores=y_scores,
        threshold=0.50,
    )

    assert "metrics" in report
    assert (tmp_path / "metrics" / "image_metrics.json").exists()
    assert (tmp_path / "figures" / "image_confusion_matrix.png").exists()
    assert (tmp_path / "figures" / "image_score_distribution.png").exists()
