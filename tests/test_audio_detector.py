"""
Unit tests for Audio Preprocessing and AudioDetector pipeline.
"""
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from src.core.config import AudioModelConfig, ThresholdConfig
from src.core.exceptions import CorruptedFileError
from src.detectors.audio import AudioDetector
from src.detectors.base import ConfidenceLevel, Verdict
from src.preprocessing.audio import AudioPreprocessor


def test_audio_preprocessor_chunk_and_pad():
    """Verify preprocessor pads short audio to 4 seconds (64,000 samples)."""
    preprocessor = AudioPreprocessor(target_sr=16000, chunk_duration_sec=4.0)

    # 1 second audio at 16kHz = 16,000 samples
    short_audio = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, 16000)).astype(np.float32)
    chunks = preprocessor.chunk_waveform(short_audio)

    assert len(chunks) == 1
    assert len(chunks[0]) == 64000  # Padded to 4s


def test_audio_preprocessor_multi_chunk_split():
    """Verify preprocessor splits long audio (9 seconds) into 3 chunks."""
    preprocessor = AudioPreprocessor(target_sr=16000, chunk_duration_sec=4.0)

    # 9 seconds = 144,000 samples -> [0:64000], [64000:128000], [128000:144000 + pad]
    long_audio = np.zeros(144000, dtype=np.float32)
    chunks = preprocessor.chunk_waveform(long_audio)

    assert len(chunks) == 3
    for c in chunks:
        assert len(c) == 64000


def test_audio_preprocessor_normalization():
    """Verify amplitude normalization prevents clipping and scales to [-1.0, 1.0]."""
    preprocessor = AudioPreprocessor()
    unnormalized = np.array([-2.5, 0.0, 5.0], dtype=np.float32)
    normalized = preprocessor.normalize(unnormalized)

    assert np.max(np.abs(normalized)) == 1.0


def test_audio_detector_classification_with_mocked_model():
    """Verify end-to-end classify() method with mocked Wav2Vec2 outputs."""
    detector = AudioDetector(
        model_config=AudioModelConfig(device="cpu"),
        threshold_config=ThresholdConfig(upper_threshold=0.65, lower_threshold=0.35),
    )

    # Mock feature extractor and model: logits [-3.0, 3.0] -> Softmax gives ~0.997 for fake (index 1)
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "real", 1: "fake"}
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[-3.0, 3.0]])
    mock_model.return_value = mock_outputs

    detector._model = mock_model
    detector._feature_extractor = MagicMock()
    detector._id2label = {0: "real", 1: "fake"}

    # Provide 1-second audio array
    dummy_audio = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, 16000)).astype(np.float32)
    
    # Mock preprocessor output
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(detector.preprocessor, "load_audio", lambda x: (dummy_audio, 16000))
        result = detector.classify(b"DUMMY_AUDIO_BYTES")

    assert result.modality == "audio"
    assert result.score > 0.90
    assert result.verdict == Verdict.LIKELY_AI
    assert result.confidence == ConfidenceLevel.HIGH
    assert "duration_seconds" in result.evidence
    assert "chunk_scores" in result.evidence


def test_audio_detector_human_classification():
    """Verify human score correctly mapped to LIKELY HUMAN-CREATED."""
    detector = AudioDetector(
        model_config=AudioModelConfig(device="cpu"),
        threshold_config=ThresholdConfig(upper_threshold=0.65, lower_threshold=0.35),
    )

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "real", 1: "fake"}
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[3.0, -3.0]])  # Softmax ~0.002 for fake
    mock_model.return_value = mock_outputs

    detector._model = mock_model
    detector._feature_extractor = MagicMock()
    detector._id2label = {0: "real", 1: "fake"}

    dummy_audio = np.sin(2 * np.pi * 200.0 * np.linspace(0, 1.0, 16000)).astype(np.float32)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(detector.preprocessor, "load_audio", lambda x: (dummy_audio, 16000))
        result = detector.classify(b"DUMMY_AUDIO_BYTES")

    assert result.score < 0.10
    assert result.verdict == Verdict.LIKELY_HUMAN
    assert result.confidence == ConfidenceLevel.HIGH
