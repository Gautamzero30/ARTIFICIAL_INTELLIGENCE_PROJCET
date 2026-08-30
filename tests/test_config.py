"""
Unit tests for configuration loading and validation.
"""
import pytest
from pathlib import Path

from src.core.config import Settings, ThresholdConfig, VideoModelConfig, load_settings
from src.core.exceptions import ValidationError


def test_default_settings():
    """Verify default settings instantiation and fields."""
    settings = Settings()
    assert settings.app.name == "Authentica AI"
    assert settings.thresholds.upper_threshold == 0.45
    assert settings.thresholds.lower_threshold == 0.40
    assert settings.image_model.primary == "umm-maybe/AI-image-detector"
    assert settings.video_model.max_duration_sec == 50


def test_threshold_validation_error():
    """Verify invalid threshold values raise ValidationError."""
    # upper < lower
    invalid_thresh = ThresholdConfig(upper_threshold=0.30, lower_threshold=0.70)
    with pytest.raises(ValidationError):
        invalid_thresh.validate()

    # out of [0, 1] range
    invalid_range = ThresholdConfig(upper_threshold=1.5, lower_threshold=0.2)
    with pytest.raises(ValidationError):
        invalid_range.validate()


def test_video_weights_validation_error():
    """Verify video fusion weights must sum to 1.0."""
    invalid_weights = VideoModelConfig(visual_weight=0.8, audio_weight=0.8)
    with pytest.raises(ValidationError):
        invalid_weights.validate()


def test_load_settings_from_real_yaml():
    """Verify load_settings correctly loads configs/config.yaml."""
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.app.version == "1.0.0"
    assert settings.thresholds.upper_threshold == 0.45
    assert settings.thresholds.lower_threshold == 0.40
    assert settings.video_model.max_duration_sec == 50
    assert ".jpg" in settings.security.allowed_extensions["image"]

