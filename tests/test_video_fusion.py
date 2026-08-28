"""
Unit tests for Video Preprocessing, Late Fusion Engine, and VideoDetector pipeline.
"""
from unittest.mock import MagicMock
import numpy as np
import pytest
from PIL import Image

from src.core.config import ThresholdConfig, VideoModelConfig
from src.core.exceptions import ProcessingError, ValidationError
from src.detectors.base import ConfidenceLevel, DetectionResult, Verdict
from src.detectors.video import VideoDetector
from src.fusion.late_fusion import FusionResult, LateFusionEngine


def test_late_fusion_dual_stream():
    """Verify dual-stream weighted fusion formula."""
    engine = LateFusionEngine(visual_weight=0.6, audio_weight=0.4)
    # Visual: 0.80, Audio: 0.60 -> Fused: (0.6 * 0.80) + (0.4 * 0.60) = 0.48 + 0.24 = 0.72
    res = engine.fuse(visual_score=0.80, audio_score=0.60)

    assert isinstance(res, FusionResult)
    assert res.fusion_mode == "dual_stream"
    assert pytest.approx(res.fused_score, 0.001) == 0.72
    assert res.visual_weight == 0.6
    assert res.audio_weight == 0.4


def test_late_fusion_silent_video_fallback():
    """Verify silent video (no audio) gracefully falls back to visual score at 100% weight."""
    engine = LateFusionEngine(visual_weight=0.6, audio_weight=0.4)
    res = engine.fuse(visual_score=0.85, audio_score=None)

    assert res.fusion_mode == "visual_only"
    assert pytest.approx(res.fused_score, 0.001) == 0.85
    assert res.visual_weight == 1.0
    assert res.audio_weight == 0.0


def test_late_fusion_audio_only_fallback():
    """Verify missing visual stream gracefully falls back to audio score at 100% weight."""
    engine = LateFusionEngine(visual_weight=0.6, audio_weight=0.4)
    res = engine.fuse(visual_score=None, audio_score=0.90)

    assert res.fusion_mode == "audio_only"
    assert pytest.approx(res.fused_score, 0.001) == 0.90
    assert res.visual_weight == 0.0
    assert res.audio_weight == 1.0


def test_late_fusion_both_failed_raises_error():
    """Verify error raised when neither visual nor audio score is provided."""
    engine = LateFusionEngine()
    with pytest.raises(ProcessingError):
        engine.fuse(visual_score=None, audio_score=None)


def test_late_fusion_invalid_weights():
    """Verify fusion weights not summing to 1.0 raise ValidationError."""
    with pytest.raises(ValidationError):
        LateFusionEngine(visual_weight=0.7, audio_weight=0.7)


def test_video_detector_classification_with_mocked_subdetectors():
    """Verify end-to-end VideoDetector classification with mocked image and audio detectors."""
    mock_img_detector = MagicMock()
    mock_img_detector.model_id = "mock-vit"
    mock_img_detector.classify.return_value = DetectionResult(
        modality="image",
        score=0.85,
        verdict=Verdict.LIKELY_AI,
        confidence=ConfidenceLevel.HIGH,
        model_name="mock-vit",
        model_version="1.0.0",
        processing_time_ms=10.0,
        disclaimer="Test",
    )

    mock_aud_detector = MagicMock()
    mock_aud_detector.model_id = "mock-wav2vec2"
    mock_aud_detector.classify.return_value = DetectionResult(
        modality="audio",
        score=0.75,
        verdict=Verdict.LIKELY_AI,
        confidence=ConfidenceLevel.MEDIUM,
        model_name="mock-wav2vec2",
        model_version="1.0.0",
        processing_time_ms=15.0,
        disclaimer="Test",
    )

    detector = VideoDetector(
        video_config=VideoModelConfig(visual_weight=0.6, audio_weight=0.4),
        threshold_config=ThresholdConfig(upper_threshold=0.65, lower_threshold=0.35),
        image_detector=mock_img_detector,
        audio_detector=mock_aud_detector,
    )

    # Mock preprocessor to return 4 dummy frames and dummy audio bytes
    dummy_frames = [Image.new("RGB", (64, 64), color="red") for _ in range(4)]
    mock_meta = {
        "total_frames": 100,
        "fps": 30.0,
        "duration_seconds": 3.33,
        "resolution": "64x64",
        "sample_timestamps": [0.0, 1.0, 2.0, 3.0],
    }

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(detector.preprocessor, "extract_keyframes", lambda x: (dummy_frames, mock_meta))
        mp.setattr(detector.preprocessor, "extract_audio_track", lambda x: (b"DUMMY_AUDIO", True))

        result = detector.classify(b"DUMMY_VIDEO_DATA")

    assert result.modality == "video"
    # Fused: (0.6 * 0.85) + (0.4 * 0.75) = 0.51 + 0.30 = 0.81
    assert pytest.approx(result.score, 0.01) == 0.81
    assert result.verdict == Verdict.LIKELY_AI
    assert result.confidence == ConfidenceLevel.MEDIUM
    assert "frame_by_frame_analysis" in result.evidence
    assert "peak_frame_anomaly" in result.evidence
    assert len(result.evidence["frame_by_frame_analysis"]) == 4
