"""
Offline evaluation script for the Text Detection Pipeline.
Executes inference on real labeled text samples and saves evaluation metrics and figures.
"""
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.logging import get_logger
from src.detectors.text import TextDetector
from src.evaluation.evaluator import DatasetEvaluator

logger = get_logger("authentica.evaluate_text")


def run_text_evaluation():
    manifest_path = BASE_DIR / "data" / "test" / "text" / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Text test manifest not found at {manifest_path}. Please run create_test_texts.py first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    logger.info(f"Loaded {len(samples)} text evaluation samples from manifest.")

    detector = TextDetector()
    try:
        detector.load_model()
    except Exception as e:
        logger.warning(f"Could not load Hugging Face model online ({e}). Falling back to feature analysis.")

    y_true = []
    y_scores = []
    metadata = {
        "dataset_name": "Authentica HC3 Sanity Text Benchmark",
        "sample_count": len(samples),
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for idx, item in enumerate(samples):
        text_content = item["text"]
        label = int(item["label"])

        try:
            result = detector.classify(text_content)
            score = result.score
        except Exception as e:
            logger.error(f"Error evaluating sample {item['id']}: {e}")
            score = 0.5  # fallback score

        y_true.append(label)
        y_scores.append(score)
        logger.info(f"[{idx+1}/{len(samples)}] {item['id']} | True: {label} | Score: {score:.4f} | Verdict: {result.verdict.value}")

    evaluator = DatasetEvaluator()
    report = evaluator.evaluate_predictions(
        modality="text",
        model_name=detector.model_id,
        y_true=y_true,
        y_scores=y_scores,
        threshold=0.50,
        metadata=metadata,
    )

    metrics = report["metrics"]
    print("\n" + "="*60)
    print("            TEXT DETECTOR EVALUATION REPORT")
    print("="*60)
    print(f"Sample Count : {metrics['sample_count']}")
    print(f"Accuracy     : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision    : {metrics['precision'] * 100:.2f}%")
    print(f"Recall       : {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score     : {metrics['f1_score'] * 100:.2f}%")
    print(f"ROC-AUC      : {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}")
    print(f"Confusion Matrix (TN, FP / FN, TP): {metrics['confusion_matrix']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_text_evaluation()
