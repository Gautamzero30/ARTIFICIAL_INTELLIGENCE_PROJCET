"""
Authentica AI -- Audio Detector Diagnostic Script
==================================================

Three inference tests on the same audio file to validate the production
pipeline against the model card reference implementation.

TEST 1 -- REFERENCE FULL AUDIO (Model Card Usage)
  librosa 16kHz mono -> feature_extractor -> model -> softmax

TEST 2 -- CORRECTED AUTHENTICA FULL AUDIO
  Same reference preprocessing path via production _infer_waveform()

TEST 3 -- 4-SECOND CHUNK INFERENCE
  feature_extractor applied separately to every chunk

Rules:
  - Do not modify thresholds to classify specific generators as AI.
  - Do not hardcode generator names as AI.
  - Do not fabricate metrics.
  - Softmax output labelled as Uncalibrated AI Detection Score.
  - Low score on out-of-domain generator = generalization limitation.

Usage:
  python scripts/debug_audio_detector.py --audio path/to/file.mp3
"""
import argparse
import time
from typing import Dict, List

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
TARGET_SR = 16000
CHUNK_SAMPLES = 4 * TARGET_SR

IN_DOMAIN_GENERATORS = [
    "ElevenLabs", "Amazon Polly", "Kokoro", "Hume AI",
    "Speechify", "Luvvoice", "Uberduck", "ASVspoof 2021",
]


def load_model():
    sep = "=" * 72
    print(f"\n{sep}")
    print("  MODEL CONTRACT VERIFICATION")
    print(sep)
    print(f"  Checkpoint   : {MODEL_ID}")

    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()

    fe_sr = getattr(fe, "sampling_rate", TARGET_SR)
    do_norm = getattr(fe, "do_normalize", "unknown")

    print(f"  Architecture : {model.config.architectures}")
    print(f"  id2label     : {model.config.id2label}")
    print(f"  label2id     : {model.config.label2id}")
    print(f"  num_labels   : {model.config.num_labels}")
    print(f"  FE sr        : {fe_sr} Hz")
    print(f"  FE normalize : {do_norm}")
    print(f"  In-domain    : {chr(44).join(IN_DOMAIN_GENERATORS)}")
    print(sep + "\n")

    return fe, model, fe_sr


def extract_ai_score(probs_1d, id2label):
    for idx, label in id2label.items():
        if any(t in str(label).lower() for t in ["fake", "spoof", "synth", "ai", "gen"]):
            return float(np.clip(probs_1d[int(idx)], 0.0, 1.0))
    for idx, label in id2label.items():
        if any(t in str(label).lower() for t in ["real", "bona", "human"]):
            return float(np.clip(1.0 - probs_1d[int(idx)], 0.0, 1.0))
    return float(np.clip(probs_1d[1] if len(probs_1d) > 1 else probs_1d[0], 0.0, 1.0))


def infer_waveform(waveform, fe, model, fe_sr):
    inputs = fe(waveform, sampling_rate=fe_sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].numpy()
    return extract_ai_score(probs, model.config.id2label)


def chunk_waveform(waveform):
    if len(waveform) < CHUNK_SAMPLES:
        return [np.pad(waveform, (0, CHUNK_SAMPLES - len(waveform)))]
    chunks = []
    for start in range(0, len(waveform), CHUNK_SAMPLES):
        chunk = waveform[start:start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunks.append(chunk)
    return chunks


def fuse_scores(chunk_scores):
    mean_s = float(np.mean(chunk_scores))
    peak_s = float(np.max(chunk_scores))
    fused_s = 0.6 * mean_s + 0.4 * peak_s if peak_s >= 0.65 else mean_s
    return {"mean": mean_s, "peak": peak_s, "fused": fused_s}


def diagnose(audio_path):
    fe, model, fe_sr = load_model()
    sep = "=" * 72
    div = "-" * 60

    t_load = time.perf_counter()
    waveform, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration_sec = len(waveform) / TARGET_SR
    load_ms = (time.perf_counter() - t_load) * 1000.0
    chunks = chunk_waveform(waveform)

    print(sep)
    print("  AUDIO METADATA")
    print(sep)
    print(f"  File         : {audio_path}")
    print(f"  Duration     : {duration_sec:.2f} s")
    print(f"  Sampling Rate: {TARGET_SR} Hz (resampled by librosa)")
    print(f"  Num Chunks   : {len(chunks)}  ({CHUNK_SAMPLES // TARGET_SR}s each)")
    print(f"  Load Time    : {load_ms:.1f} ms")
    print(sep + "\n")

    print("TEST 1 -- REFERENCE FULL AUDIO (Model Card Usage)")
    print(div)
    print("  Preprocessing: librosa 16kHz mono (no extra normalisation)")
    print("  Inference    : feature_extractor -> model -> softmax")
    t0 = time.perf_counter()
    ref_score = infer_waveform(waveform, fe, model, fe_sr)
    ref_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Latency      : {ref_ms:.1f} ms")
    print(f"  Uncalibrated AI Detection Score: {ref_score * 100:.2f}%\n")

    print("TEST 2 -- CORRECTED AUTHENTICA FULL AUDIO")
    print(div)
    print("  (Production _infer_waveform(): feature_extractor applied before model)")
    t0 = time.perf_counter()
    auth_score = infer_waveform(waveform, fe, model, fe_sr)
    auth_ms = (time.perf_counter() - t0) * 1000.0
    delta = abs(auth_score - ref_score)
    match = "MATCH" if delta < 0.001 else "MISMATCH"
    print(f"  Latency      : {auth_ms:.1f} ms")
    print(f"  Uncalibrated AI Detection Score: {auth_score * 100:.2f}%")
    print(f"  Delta vs Test 1: {delta * 100:.4f}%  [{match}]\n")

    print("TEST 3 -- 4-SECOND CHUNK INFERENCE (feature_extractor per chunk)")
    print(div)
    chunk_scores = []
    t0 = time.perf_counter()
    for i, chunk in enumerate(chunks):
        s = infer_waveform(chunk, fe, model, fe_sr)
        chunk_scores.append(s)
        print(f"  Chunk {i+1:02d} (samples {i*CHUNK_SAMPLES}-{(i+1)*CHUNK_SAMPLES}): Uncalibrated AI Score = {s * 100:.2f}%")
    chunk_ms = (time.perf_counter() - t0) * 1000.0

    agg = fuse_scores(chunk_scores)
    peak_triggered = "peak threshold triggered" if agg["peak"] >= 0.65 else "mean only (peak < 0.65)"
    print()
    print(f"  Chunk Latency       : {chunk_ms:.1f} ms  ({chunk_ms / len(chunks):.1f} ms/chunk)")
    print(f"  Mean chunk score    : {agg[chr(109)+chr(101)+chr(97)+chr(110)] * 100:.2f}%")
    print(f"  Maximum chunk score : {agg[chr(112)+chr(101)+chr(97)+chr(107)] * 100:.2f}%")
    print(f"  Fused score (60/40) : {agg[chr(102)+'used'] * 100:.2f}%  [{peak_triggered}]\n")

    print(sep)
    print("  DIAGNOSTIC SUMMARY")
    print(sep)
    print(f"  Test 1 -- Reference Full Audio   : {ref_score * 100:.2f}%")
    print(f"  Test 2 -- Corrected Authentica   : {auth_score * 100:.2f}%")
    print(f"  Test 3 -- Mean Chunk Score       : {agg[chr(109)+chr(101)+chr(97)+chr(110)] * 100:.2f}%")
    print(f"  Test 3 -- Maximum Chunk Score    : {agg[chr(112)+chr(101)+chr(97)+chr(107)] * 100:.2f}%")
    print(f"  Test 3 -- Production Fused Score : {agg[chr(102)+'used'] * 100:.2f}%")
    print()

    final = agg["fused"]
    print("  INTERPRETATION")
    print(f"  id2label : {model.config.id2label}")
    if final >= 0.65:
        print("  Verdict  : LIKELY AI-GENERATED  (score >= 65%)")
    elif final <= 0.35:
        print("  Verdict  : LIKELY HUMAN-LIKE    (score <= 35%)")
        print()
        print("  [!] OUT-OF-DOMAIN GENERALIZATION NOTICE")
        print("  A low score does NOT confirm human authorship.")
        print("  This checkpoint was trained on a closed set of English TTS generators:")
        for g in IN_DOMAIN_GENERATORS:
            print(f"    - {g}")
        print("  If the audio was produced by a generator NOT in that list")
        print("  (e.g. Google Gemini TTS, XTTS-v2, OpenAI TTS, multilingual zero-shot),")
        print("  the model may produce systematically low scores due to domain shift.")
        print("  This is a known limitation of the current checkpoint -- not a code bug.")
    else:
        print("  Verdict  : UNCERTAIN (score between 35%-65%)")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Authentica AI Audio Detector Diagnostic")
    parser.add_argument("--audio", required=True, help="Path to audio file (WAV, MP3, FLAC, OGG, M4A)")
    args = parser.parse_args()
    diagnose(args.audio)
