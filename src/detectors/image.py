"""
Image Detection pipeline for Authentica AI using Vision Transformers.
"""
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.core.config import ImageModelConfig, ThresholdConfig
from src.core.exceptions import InferenceError, ModelLoadError
from src.core.logging import get_logger
from src.detectors.base import BaseDetector, DetectionResult, compute_verdict_and_confidence
from src.preprocessing.image import ImagePreprocessor

logger = get_logger("authentica.image_detector")


class ImageDetector(BaseDetector):
    """
    Vision Transformer AI-Image Detector.
    Estimates whether an input image is AI-generated (e.g. Stable Diffusion, Midjourney, DALL-E)
    or human-created photograph.
    """
    MODALITY = "image"

    def __init__(
        self,
        model_config: Optional[ImageModelConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None,
    ):
        self.config = model_config or ImageModelConfig()
        super().__init__(
            model_id=self.config.primary,
            threshold_cfg=threshold_config,
        )
        self.preprocessor = ImagePreprocessor(target_size=self.config.input_size)
        self.device = self._resolve_device(self.config.device)
        self._model = None
        self._id2label = {}

    def _resolve_device(self, requested_device: str) -> torch.device:
        """Determines computation device based on user preference and hardware."""
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested_device)

    def load_model(self) -> None:
        """
        Loads the Vision Transformer classification model from Hugging Face.
        """
        if self._model is not None:
            return

        try:
            logger.info(f"Loading image detection model '{self.model_id}' on device '{self.device}'...")
            from transformers import AutoModelForImageClassification

            self._model = AutoModelForImageClassification.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()

            # Cache label mapping
            if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                self._id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
            else:
                # Default binary assumption: 0=artificial, 1=human
                self._id2label = {0: "artificial", 1: "human"}

            logger.info(f"Image detection model loaded successfully. Label map: {self._id2label}")

        except Exception as e:
            logger.error(f"Failed to load image model '{self.model_id}': {e}")
            raise ModelLoadError(f"Could not load image detection model '{self.model_id}': {e}")

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """
        Preprocesses raw image input into a normalized tensor.
        """
        return self.preprocessor.preprocess(raw_input)

    def _extract_ai_probability(self, probs: torch.Tensor) -> float:
        """
        Extracts AI probability score S_AI from output softmax probabilities
        based on the model's id2label configuration.
        """
        probs_np = probs.cpu().detach().numpy()[0]

        # 1. Look for explicit AI/fake/artificial label index
        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["art", "fake", "ai", "synth", "gen"]):
                return float(probs_np[idx])

        # 2. Look for human/real label and invert
        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["hum", "real"]):
                return float(1.0 - probs_np[idx])

        # Fallback to index 0
        return float(probs_np[0])

    def predict(self, preprocessed_input: torch.Tensor) -> float:
        """
        Executes model forward pass and returns uncalibrated AI score in [0.0, 1.0].
        """
        self.load_model()

        try:
            tensor = preprocessed_input.to(self.device)
            with torch.no_grad():
                outputs = self._model(tensor)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)

            ai_score = self._extract_ai_probability(probs)
            # Ensure clamped within [0.0, 1.0]
            return max(0.0, min(1.0, ai_score))

        except Exception as e:
            logger.error(f"Error during image inference: {e}")
            raise InferenceError(f"Image detection inference failed: {e}")

    def classify(self, raw_input: Any) -> DetectionResult:
        """
        Complete image analysis pipeline with timing and structured evidence.
        """
        import time
        start_time = time.perf_counter()

        # Extract basic image metadata before tensor conversion
        pil_img = self.preprocessor.load_image(raw_input)
        original_size = pil_img.size

        preprocessed_tensor = self.preprocessor.preprocess(pil_img)
        ai_score = self.predict(preprocessed_tensor)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        verdict, confidence = compute_verdict_and_confidence(ai_score, self.threshold_cfg)

        return DetectionResult(
            modality=self.MODALITY,
            score=ai_score,
            verdict=verdict,
            confidence=confidence,
            model_name=self.model_id,
            model_version="1.0.0",
            processing_time_ms=elapsed_ms,
            disclaimer="This result is an AI-based estimate and is not definitive proof of whether the content was generated by AI.",
            is_calibrated=False,
            evidence={
                "raw_ai_score": round(ai_score, 4),
                "human_score": round(1.0 - ai_score, 4),
                "original_dimensions": f"{original_size[0]}x{original_size[1]}",
                "processed_resolution": f"{self.config.input_size[0]}x{self.config.input_size[1]}",
                "device": str(self.device),
            },
        )
