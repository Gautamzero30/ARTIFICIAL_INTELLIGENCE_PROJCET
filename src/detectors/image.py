"""
Image Detection pipeline for Authentica AI using Vision Transformers and High-Frequency Residual Analysis.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
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
    Produces uncalibrated AI Likeness Detection Scores.
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
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested_device)

    def load_model(self) -> None:
        if self._model is not None:
            return

        try:
            logger.info(f"Loading image detection model '{self.model_id}' on device '{self.device}'...")
            from transformers import AutoModelForImageClassification

            self._model = AutoModelForImageClassification.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()

            if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                self._id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
            else:
                self._id2label = {0: "artificial", 1: "human"}

            logger.info(f"Image detection model loaded. Label map: {self._id2label}")

        except Exception as e:
            logger.error(f"Failed to load image model '{self.model_id}': {e}")
            raise ModelLoadError(f"Could not load image detection model '{self.model_id}': {e}")

    def analyze_frequency_residuals(self, pil_img: Image.Image) -> Dict[str, Any]:
        """
        Extracts mathematical Fourier spectral residuals and Laplacian edge variance.
        """
        img_np = np.array(pil_img.convert("L"))

        # 1. Laplacian Edge Sharpness / Variance
        laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
        laplacian_var = float(np.var(laplacian))

        # 2. 2D FFT High-Frequency Energy Ratio
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        h, w = img_np.shape
        cy, cx = h // 2, w // 2
        r_inner = min(h, w) // 6
        y, x = np.ogrid[:h, :w]
        mask_inner = (x - cx)**2 + (y - cy)**2 <= r_inner**2

        low_freq_energy = float(np.mean(magnitude_spectrum[mask_inner]))
        high_freq_energy = float(np.mean(magnitude_spectrum[~mask_inner]))
        freq_ratio = float(high_freq_energy / (low_freq_energy + 1e-6))

        return {
            "laplacian_variance": round(laplacian_var, 2),
            "high_low_frequency_ratio": round(freq_ratio, 3),
            "texture_smoothness": "High" if laplacian_var < 150 else "Standard Texture",
            "spectrum_anomaly_detected": bool(freq_ratio < 0.65 or freq_ratio > 1.35),
        }

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        return self.preprocessor.preprocess(raw_input)

    def _extract_ai_probability(self, probs: torch.Tensor) -> float:
        probs_np = probs.cpu().detach().numpy()[0]

        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["art", "fake", "ai", "synth", "gen"]):
                return float(probs_np[idx])

        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["hum", "real"]):
                return float(1.0 - probs_np[idx])

        return float(probs_np[0])

    def predict(self, preprocessed_input: torch.Tensor) -> float:
        self.load_model()

        try:
            tensor = preprocessed_input.to(self.device)
            with torch.no_grad():
                outputs = self._model(tensor)
                probs = F.softmax(outputs.logits, dim=-1)

            ai_score = self._extract_ai_probability(probs)
            return max(0.0, min(1.0, ai_score))

        except Exception as e:
            logger.error(f"Error during image inference: {e}")
            raise InferenceError(f"Image detection inference failed: {e}")

    def classify(self, raw_input: Any) -> DetectionResult:
        import time
        start_time = time.perf_counter()

        pil_img = self.preprocessor.load_image(raw_input)
        original_size = pil_img.size

        # High-frequency forensic metrics
        freq_forensics = self.analyze_frequency_residuals(pil_img)

        preprocessed_tensor = self.preprocessor.preprocess(pil_img)
        ai_score = self.predict(preprocessed_tensor)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        verdict, confidence = compute_verdict_and_confidence(ai_score, self.threshold_cfg)

        # Strictly factual evidence (no fabricated assertions)
        score_pct = ai_score * 100.0
        factors = [
            f"Vision Transformer Model Classification Score: {score_pct:.1f}% AI Likeness",
        ]
        if ai_score > self.threshold_cfg.upper_threshold:
            factors.append("Feature representations align with the detector's synthetic training distribution")
            if freq_forensics["spectrum_anomaly_detected"]:
                factors.append("Observed high-frequency Fourier spectral decay anomaly")
        elif ai_score < self.threshold_cfg.lower_threshold:
            factors.append("Feature representations align with the detector's authentic photographic training distribution")
        else:
            factors.append("Classification score falls within the uncertain decision band (40%–45%)")

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
                "dimensions": f"{original_size[0]} x {original_size[1]} px",
                "aspect_ratio": f"{original_size[0]/max(1, original_size[1]):.2f}:1",
                "forensic_factors": factors,
                "frequency_analysis": freq_forensics,
                "device": str(self.device),
            },
        )
