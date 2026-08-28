"""
Unit tests for Text Preprocessing and TextDetector pipeline.
"""
from unittest.mock import MagicMock
import pytest
import torch

from src.core.config import TextModelConfig, ThresholdConfig
from src.core.exceptions import InsufficientContentError
from src.detectors.base import ConfidenceLevel, Verdict
from src.detectors.text import TextDetector
from src.preprocessing.text import TextPreprocessor


def test_text_cleaner_and_normalizer():
    """Verify text cleaner removes excess whitespace and normalizes Unicode."""
    preprocessor = TextPreprocessor(min_characters=50)
    dirty_text = "  This is   a  sample   test  paragraph    with multiple  spaces and newlines.\n\n\n\nIt should be cleaned.  "
    cleaned = preprocessor.clean_text(dirty_text)
    
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("This is")
    assert cleaned.endswith("cleaned.")


def test_empty_text_raises_insufficient_content():
    """Verify empty or whitespace-only text raises InsufficientContentError."""
    preprocessor = TextPreprocessor(min_characters=50)
    with pytest.raises(InsufficientContentError):
        preprocessor.validate_text("")

    with pytest.raises(InsufficientContentError):
        preprocessor.validate_text("   \n\t  ")


def test_short_text_raises_insufficient_content():
    """Verify short text under 50 characters raises InsufficientContentError."""
    preprocessor = TextPreprocessor(min_characters=50)
    short_text = "This text is too short."
    with pytest.raises(InsufficientContentError):
        preprocessor.validate_text(short_text)


def test_non_english_script_detection():
    """Verify detector identifies non-Latin/multilingual text."""
    preprocessor = TextPreprocessor(min_characters=10)
    english_text = "This is a purely English sentence with standard ASCII characters."
    nepali_text = "यो नेपाली भाषामा लेखिएको एउटा परीक्षण पाठ हो।"

    assert preprocessor.detect_non_english(english_text) is False
    assert preprocessor.detect_non_english(nepali_text) is True


def test_text_detector_classification_with_mocked_model():
    """Verify end-to-end classify() method with mocked RoBERTa outputs."""
    detector = TextDetector(
        model_config=TextModelConfig(device="cpu", min_character_length=50),
        threshold_config=ThresholdConfig(upper_threshold=0.65, lower_threshold=0.35),
    )

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.ones((1, 100), dtype=torch.long),
        "attention_mask": torch.ones((1, 100), dtype=torch.long),
    }
    # Mock model: logits [0.0, 3.0] -> Softmax gives ~0.95 for ChatGPT (index 1)
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "Human", 1: "ChatGPT"}
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.0, 3.0]])
    mock_model.return_value = mock_outputs

    detector._tokenizer = mock_tokenizer
    detector._model = mock_model
    detector._id2label = {0: "human", 1: "chatgpt"}

    valid_sample = (
        "Artificial intelligence and deep learning models are advancing at an exponential pace. "
        "Language models such as GPT-4 can generate essays that resemble human writing style."
    )

    result = detector.classify(valid_sample)

    assert result.modality == "text"
    assert result.score > 0.90
    assert result.verdict == Verdict.LIKELY_AI
    assert result.confidence == ConfidenceLevel.HIGH
    assert "word_count" in result.evidence
    assert "character_count" in result.evidence
    assert result.evidence["word_count"] > 10


def test_text_detector_human_classification():
    """Verify human score correctly mapped to LIKELY HUMAN-CREATED."""
    detector = TextDetector(
        model_config=TextModelConfig(device="cpu", min_character_length=50),
        threshold_config=ThresholdConfig(upper_threshold=0.65, lower_threshold=0.35),
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.ones((1, 50), dtype=torch.long),
        "attention_mask": torch.ones((1, 50), dtype=torch.long),
    }
    # Logits [3.0, 0.0] -> Softmax gives ~0.05 for AI (index 1)
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "Human", 1: "ChatGPT"}
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[3.0, 0.0]])
    mock_model.return_value = mock_outputs

    detector._tokenizer = mock_tokenizer
    detector._model = mock_model
    detector._id2label = {0: "human", 1: "chatgpt"}

    valid_sample = (
        "Yesterday I went to the grocery store to buy fresh fruits, vegetables, and milk for dinner. "
        "The weather was remarkably pleasant and cool throughout the evening."
    )

    result = detector.classify(valid_sample)

    assert result.score < 0.10
    assert result.verdict == Verdict.LIKELY_HUMAN
    assert result.confidence == ConfidenceLevel.HIGH
