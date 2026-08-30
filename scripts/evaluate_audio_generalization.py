"""
Offline Generator Generalization & Ablation Evaluation Suite for Audio Detection.
Evaluates in-domain (seen generators) vs out-of-domain (unseen generators) and compares
aggregation strategies (Full Audio, 4s Mean, 4s Max, 0.60/0.40 Fusion).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.config import load_settings
from src.detectors.audio import AudioDetector


def calculate_metrics(y_true: List[int], y_scores: List[float], threshold: float = 0.50) -> Dict[str, Any]:
    """Calculates empirical evaluation metrics without fabrication."""
    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)
    y_pred_arr = (y_scores_arr >= threshold).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    
    # Handle single-class scenarios gracefully
    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) > 1:
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
        except Exception:
            roc_auc = None
        try:
            pr_auc = float(average_precision_score(y_true_arr, y_scores_arr))
        except Exception:
            pr_auc = None
        cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()
    else:
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        roc_auc = None
        pr_auc = None
        cm = [[int(np.sum(y_pred_arr == 0)), int(np.sum(y_pred_arr == 1))]]

    return {
        "sample_count": len(y_true),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": cm,
    }


def evaluate_audio_dataset(manifest_path: Path) -> Dict[str, Any]:
    """Runs ablation & domain evaluation across audio samples in manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    detector = AudioDetector()
    detector.load_model()

    results_by_strategy = {
        "reference_full": {"y_true": [], "y_scores": []},
        "chunk_mean": {"y_true": [], "y_scores": []},
        "chunk_max": {"y_true": [], "y_scores": []},
        "fused_0.6_0.4": {"y_true": [], "y_scores": []},
    }

    in_domain_data = {"y_true": [], "y_scores": []}
    out_of_domain_data = {"y_true": [], "y_scores": []}

    print(f"Evaluating {len(samples)} audio samples...")
    for idx, item in enumerate(samples):
        filepath = Path(item["path"])
        label = int(item["label"])
        generator = item.get("generator", item.get("source", "unknown")).lower()
        is_in_domain = any(g in generator for g in ["asvspoof", "elevenlabs", "polly", "tacotron", "vits", "natural"])

        if not filepath.exists():
            continue

        try:
            # 1. Reference full inference
            y, sr = librosa.load(str(filepath), sr=16000, mono=True)
            inputs = detector._feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                out = detector._model(**inputs)
                p = F.softmax(out.logits, dim=-1)
                score_ref = detector._extract_ai_probability(p)

            # 2. Pipeline chunk inference
            with open(filepath, "rb") as af:
                raw_bytes = af.read()
            chunks, _ = detector.preprocessor.preprocess(raw_bytes)
            c_scores = []
            with torch.no_grad():
                for c in chunks:
                    c_inputs = c.to(detector.device)
                    c_out = detector._model(c_inputs)
                    c_probs = F.softmax(c_out.logits, dim=-1)
                    c_scores.append(detector._extract_ai_probability(c_probs))

            score_mean = float(np.mean(c_scores)) if c_scores else score_ref
            score_max = float(np.max(c_scores)) if c_scores else score_ref
            score_fused = (0.60 * score_mean + 0.40 * score_max) if (c_scores and score_max >= 0.65) else score_mean

            # Collect results
            results_by_strategy["reference_full"]["y_true"].append(label)
            results_by_strategy["reference_full"]["y_scores"].append(score_ref)

            results_by_strategy["chunk_mean"]["y_true"].append(label)
            results_by_strategy["chunk_mean"]["y_scores"].append(score_mean)

            results_by_strategy["chunk_max"]["y_true"].append(label)
            results_by_strategy["chunk_max"]["y_scores"].append(score_max)

            results_by_strategy["fused_0.6_0.4"]["y_true"].append(label)
            results_by_strategy["fused_0.6_0.4"]["y_scores"].append(score_fused)

            if is_in_domain:
                in_domain_data["y_true"].append(label)
                in_domain_data["y_scores"].append(score_fused)
            else:
                out_of_domain_data["y_true"].append(label)
                out_of_domain_data["y_scores"].append(score_fused)

        except Exception as e:
            print(f"Error on {filepath.name}: {e}")

    report = {
        "model_id": detector.model_id,
        "sample_count": len(samples),
        "ablation_metrics": {},
        "domain_generalization": {},
    }

    print("\n" + "=" * 80)
    print("AUDIO DETECTOR ABLATION COMPARISON (On Held-Out Evaluation Dataset)")
    print("=" * 80)
    print(f"{'Strategy':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 80)

    for name, data in results_by_strategy.items():
        m = calculate_metrics(data["y_true"], data["y_scores"])
        report["ablation_metrics"][name] = m
        roc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        print(f"{name:<30} | {m['accuracy']*100:6.2f}%   | {m['precision']*100:6.2f}%   | {m['recall']*100:6.2f}%   | {m['f1_score']*100:6.2f}%   | {roc_str:<10}")

    print("\n" + "=" * 80)
    print("DOMAIN GENERALIZATION BREAKDOWN")
    print("=" * 80)

    if in_domain_data["y_true"]:
        m_in = calculate_metrics(in_domain_data["y_true"], in_domain_data["y_scores"])
        report["domain_generalization"]["in_domain_seen_generators"] = m_in
        print(f"In-Domain (Seen Generators: ASVspoof/ElevenLabs/Polly) : N={m_in['sample_count']}, Acc={m_in['accuracy']*100:.1f}%, F1={m_in['f1_score']*100:.1f}%")
    else:
        print("In-Domain: No samples categorized")

    if out_of_domain_data["y_true"]:
        m_out = calculate_metrics(out_of_domain_data["y_true"], out_of_domain_data["y_scores"])
        report["domain_generalization"]["out_of_domain_unseen_generators"] = m_out
        print(f"Out-of-Domain (Unseen Generators: Gemini/Zero-Shot TTS)  : N={m_out['sample_count']}, Acc={m_out['accuracy']*100:.1f}%, F1={m_out['f1_score']*100:.1f}%")
    else:
        print("Out-of-Domain: No samples categorized")

    print("=" * 80 + "\n")
    return report


if __name__ == "__main__":
    manifest_file = BASE_DIR / "data" / "test" / "audio" / "manifest.json"
    evaluate_audio_dataset(manifest_file)
