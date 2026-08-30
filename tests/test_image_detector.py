"""
Unit tests for the Image Preprocessing and ImageDetector pipeline.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
from PIL import Image

from src.core.config import ImageModelConfig, ThresholdConfig
from src.core.exceptions import CorruptedFileError
from src.detectors.base import ConfidenceLevel, Verdict
from src.detectors.image import ImageDetector
from src.preprocessing.image import ImagePreprocessor


def create_test_pil_image(mode: str = "RGB", size=(100, 100)) -> Image.Image:
    """Helper to generate dummy PIL images."""
    if mode == "RGB":
        data = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        return Image.fromarray(data).convert("RGB")
    elif mode == "RGBA":
        data = np.random.randint(0, 256, (size[1], size[0], 4), dtype=np.uint8)
        return Image.fromarray(data).convert("RGBA")
    elif mode == "L":
        data = np.random.randint(0, 256, (size[1], size[0]), dtype=np.uint8)
        return Image.fromarray(data).convert("L")
    raise ValueError(f"Unsupported mode: {mode}")


def test_image_preprocessor_load_and_convert_modes():
    """Verify preprocessor converts RGBA and Grayscale to RGB cleanly."""
    preprocessor = ImagePreprocessor(target_size=(224, 224))

    # Test RGB
    rgb_img = create_test_pil_image("RGB")
    loaded_rgb = preprocessor.load_image(rgb_img)
    assert loaded_rgb.mode == "RGB"

    # Test RGBA
    rgba_img = create_test_pil_image("RGBA")
    loaded_rgba = preprocessor.load_image(rgba_img)
    assert loaded_rgba.mode == "RGB"

    # Test Grayscale
    gray_img = create_test_pil_image("L")
    loaded_gray = preprocessor.load_image(gray_img)
    assert loaded_gray.mode == "RGB"


def test_image_preprocessor_output_tensor_shape():
    """Verify tensor shape is [1, 3, 224, 224] with normalization applied."""
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    img = create_test_pil_image("RGB", size=(400, 300))
    tensor = preprocessor.preprocess(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_image_preprocessor_corrupted_input():
    """Verify corrupted bytes raise CorruptedFileError."""
    preprocessor = ImagePreprocessor()
    with pytest.raises(CorruptedFileError):
        preprocessor.load_image(b"INVALID_IMAGE_PAYLOAD")


def test_image_detector_classification_with_mocked_model():
    """Verify end-to-end classify() method with mocked ViT outputs."""
    detector = ImageDetector(
        model_config=ImageModelConfig(device="cpu"),
        threshold_config=ThresholdConfig(upper_threshold=0.45, lower_threshold=0.40),
    )

    # Mock HF AutoModel
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "artificial", 1: "human"}

    # Mock logits where AI (index 0) has high score: 0.90 AI, 0.10 Human
    # Logits correspond to approx softmax [0.90, 0.10] -> [2.197, 0.0]
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[2.197, 0.0]])
    mock_model.return_value = mock_outputs

    detector._model = mock_model
    detector._id2label = {0: "artificial", 1: "human"}

    test_img = create_test_pil_image("RGB", size=(150, 150))
    result = detector.classify(test_img)

    assert result.modality == "image"
    assert result.score > 0.85
    assert result.verdict == Verdict.LIKELY_AI
    assert result.confidence == ConfidenceLevel.HIGH
    assert "raw_ai_score" in result.evidence
    assert "human_score" in result.evidence
    assert result.is_calibrated is False


def test_image_detector_human_classification():
    """Verify human score correctly mapped to LIKELY HUMAN-CREATED."""
    detector = ImageDetector(
        model_config=ImageModelConfig(device="cpu"),
        threshold_config=ThresholdConfig(upper_threshold=0.45, lower_threshold=0.40),
    )


    mock_model = MagicMock()
    mock_model.config.id2label = {0: "artificial", 1: "human"}
    # Logits for 0.05 AI, 0.95 Human
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[-2.944, 0.0]])
    mock_model.return_value = mock_outputs

    detector._model = mock_model
    detector._id2label = {0: "artificial", 1: "human"}

    test_img = create_test_pil_image("RGB", size=(150, 150))
    result = detector.classify(test_img)

    assert result.score < 0.15
    assert result.verdict == Verdict.LIKELY_HUMAN
    assert result.confidence == ConfidenceLevel.HIGH
