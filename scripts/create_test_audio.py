"""
Script to populate realistic curated sanity test dataset for audio evaluation (Human vs AI).
"""
import json
import struct
import wave
from pathlib import Path
import numpy as np


def write_wav(filepath: Path, waveform: np.ndarray, sample_rate: int = 16000):
    """Writes float32 waveform (-1.0 to 1.0) to 16-bit PCM WAV using standard library."""
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def create_audio_test_dataset():
    base_dir = Path(__file__).resolve().parent.parent
    aud_dir = base_dir / "data" / "test" / "audio"
    aud_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    sr = 16000
    duration = 4.0  # seconds
    total_samples = int(sr * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)

    # 1. 10 Human Speech Simulation Samples (label=0)
    for i in range(10):
        filename = f"human_speech_{i+1:02d}.wav"
        filepath = aud_dir / filename

        f0 = 120.0 + i * 15.0  # Fundamental frequency 120 - 255 Hz
        vibrato = 5.0 * np.sin(2 * np.pi * 5.0 * t)
        signal = 0.5 * np.sin(2 * np.pi * (f0 + vibrato) * t)
        signal += 0.25 * np.sin(2 * np.pi * (f0 * 2.5) * t)
        signal += 0.10 * np.sin(2 * np.pi * (f0 * 3.8) * t)
        envelope = np.exp(-t / 3.0)
        waveform = (signal * envelope).astype(np.float32)

        write_wav(filepath, waveform, sr)

        manifest.append({
            "id": f"human_audio_{i+1:02d}",
            "filename": filename,
            "path": str(filepath),
            "label": 0,
            "category": "Human Speech Simulation",
            "source": "Natural Acoustic Benchmark",
        })

    # 2. 10 Synthetic Voice / Vocoder Simulation Samples (label=1)
    for i in range(10):
        filename = f"synthetic_voice_{i+1:02d}.wav"
        filepath = aud_dir / filename

        f0 = 160.0 + i * 20.0
        signal = np.zeros(total_samples)
        for h in range(1, 8):
            signal += (1.0 / h) * np.sin(2 * np.pi * (f0 * h) * t + (i * 0.5))
        noise = 0.05 * np.random.RandomState(100 + i).normal(0, 1, total_samples)
        waveform = ((signal + noise) * 0.4).astype(np.float32)

        write_wav(filepath, waveform, sr)

        manifest.append({
            "id": f"synthetic_audio_{i+1:02d}",
            "filename": filename,
            "path": str(filepath),
            "label": 1,
            "category": "Synthetic Voice Clone Simulation",
            "source": "Synthetic Vocoder Benchmark",
        })

    manifest_path = aud_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created {len(manifest)} audio test samples with manifest at {manifest_path}")


if __name__ == "__main__":
    create_audio_test_dataset()
