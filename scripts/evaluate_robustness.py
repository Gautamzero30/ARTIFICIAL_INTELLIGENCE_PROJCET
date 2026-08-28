"""
Robustness evaluation script for measuring performance under realistic real-world distortions.
"""
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.logging import get_logger
from src.detectors.image import ImageDetector
from src.detectors.text import TextDetector
from src.evaluation.metrics import compute_binary_metrics

logger = get_logger("authentica.evaluate_robustness")


def evaluate_image_jpeg_compression_robustness(detector: ImageDetector):
    manifest_path = BASE_DIR / "data" / "test" / "image" / "manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    y_true = []
    y_scores_orig = []
    y_scores_compressed = []

    for item in samples:
        filepath = Path(item["path"])
        label = int(item["label"])
        y_true.append(label)

        # 1. Original
        res_orig = detector.classify(filepath)
        y_scores_orig.append(res_orig.score)

        # 2. Perturbed: Severe JPEG Compression (Q=30)
        img = Image.open(filepath).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        res_comp = detector.classify(buf.getvalue())
        y_scores_compressed.append(res_comp.score)

    m_orig = compute_binary_metrics(y_true, y_scores_orig, threshold=0.50)
    m_comp = compute_binary_metrics(y_true, y_scores_compressed, threshold=0.50)

    return {
        "perturbation": "JPEG Compression (Quality=30)",
        "sample_count": len(samples),
        "original_accuracy": m_orig.accuracy,
        "compressed_accuracy": m_comp.accuracy,
        "original_f1": m_orig.f1_score,
        "compressed_f1": m_comp.f1_score,
        "original_roc_auc": m_orig.roc_auc,
        "compressed_roc_auc": m_comp.roc_auc,
    }


def evaluate_text_truncation_robustness(detector: TextDetector):
    manifest_path = BASE_DIR / "data" / "test" / "text" / "manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    y_true = []
    y_scores_orig = []
    y_scores_trunc = []

    for item in samples:
        text = item["text"]
        label = int(item["label"])
        y_true.append(label)

        # Original
        res_orig = detector.classify(text)
        y_scores_orig.append(res_orig.score)

        # Truncated to first 60 characters
        truncated_text = text[:min(len(text), 65)]
        try:
            res_trunc = detector.classify(truncated_text)
            y_scores_trunc.append(res_trunc.score)
        except Exception:
            y_scores_trunc.append(0.5)

    m_orig = compute_binary_metrics(y_true, y_scores_orig, threshold=0.50)
    m_trunc = compute_binary_metrics(y_true, y_scores_trunc, threshold=0.50)

    return {
        "perturbation": "Text Truncation (Shortened to ~65 chars)",
        "sample_count": len(samples),
        "original_accuracy": m_orig.accuracy,
        "truncated_accuracy": m_trunc.accuracy,
        "original_f1": m_orig.f1_score,
        "truncated_f1": m_trunc.f1_score,
        "original_roc_auc": m_orig.roc_auc,
        "truncated_roc_auc": m_trunc.roc_auc,
    }


def run_robustness_evaluation():
    logger.info("Executing robustness perturbation experiments...")
    exp_dir = BASE_DIR / "reports" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    img_det = ImageDetector()
    img_results = evaluate_image_jpeg_compression_robustness(img_det)

    txt_det = TextDetector()
    txt_results = evaluate_text_truncation_robustness(txt_det)

    results_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_robustness": img_results,
        "text_robustness": txt_results,
    }

    out_file = exp_dir / "robustness_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    logger.info(f"Robustness results successfully saved to {out_file}")
    print("\n" + "="*60)
    print("            ROBUSTNESS EXPERIMENT SUMMARY")
    print("="*60)
    if img_results:
        print(f"Image Original Accuracy    : {img_results['original_accuracy']*100:.1f}%")
        print(f"Image Post-JPEG Accuracy   : {img_results['compressed_accuracy']*100:.1f}%")
    if txt_results:
        print(f"Text Original Accuracy     : {txt_results['original_accuracy']*100:.1f}%")
        print(f"Text Post-Trunc Accuracy   : {txt_results['truncated_accuracy']*100:.1f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_robustness_evaluation()
