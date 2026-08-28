"""
Synthetic Audio & Voice Clone Detection pipeline for Authentica AI using Wav2Vec2.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.core.config import AudioModelConfig, ThresholdConfig
from src.core.exceptions import InferenceError, ModelLoadError
from src.core.logging import get_logger
from src.detectors.base import BaseDetector, DetectionResult, compute_verdict_and_confidence
from src.preprocessing.audio import AudioPreprocessor

logger = get_logger("authentica.audio_detector")


class AudioDetector(BaseDetector):
    """
    Wav2Vec2 Synthetic Audio & Voice Clone Detector.
    Estimates whether input speech is AI-generated/cloned (e.g. ElevenLabs, Tacotron, VITS)
    or authentic human speech.
    """
    MODALITY = "audio"

    def __init__(
        self,
        model_config: Optional[AudioModelConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None,
    ):
        self.config = model_config or AudioModelConfig()
        super().__init__(
            model_id=self.config.primary,
            threshold_cfg=threshold_config,
        )
        self.preprocessor = AudioPreprocessor(
            target_sr=self.config.target_sample_rate,
            chunk_duration_sec=self.config.chunk_duration_sec,
        )
        self.device = self._resolve_device(self.config.device)
        self._model = None
        self._feature_extractor = None
        self._id2label = {}

    def _resolve_device(self, requested_device: str) -> torch.device:
        """Determines computation device based on user preference and hardware."""
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested_device)

    def load_model(self) -> None:
        """
        Loads the Wav2Vec2 classification model and feature extractor from Hugging Face.
        """
        if self._model is not None and self._feature_extractor is not None:
            return

        try:
            logger.info(f"Loading audio detection model '{self.model_id}' on device '{self.device}'...")
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()

            # Cache label mapping
            if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                self._id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
            else:
                # Default binary assumption: 0=real, 1=fake/synthetic
                self._id2label = {0: "real", 1: "fake"}

            logger.info(f"Audio detection model loaded successfully. Label map: {self._id2label}")

        except Exception as e:
            logger.error(f"Failed to load audio model '{self.model_id}': {e}")
            raise ModelLoadError(f"Could not load audio detection model '{self.model_id}': {e}")

    def preprocess(self, raw_input: Any) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Loads and converts audio into normalized 16kHz chunk tensors.
        """
        return self.preprocessor.preprocess(raw_input)

    def _extract_ai_probability(self, probs: torch.Tensor) -> float:
        """
        Extracts AI/fake probability score S_AI from output softmax probabilities.
        """
        probs_np = probs.cpu().detach().numpy()[0]

        # 1. Look for explicit fake / spoof / synth / ai label index
        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["fake", "spoof", "synth", "ai", "gen"]):
                return float(probs_np[idx])

        # 2. Look for real / bonafide / human label and invert
        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["real", "bona", "human"]):
                return float(1.0 - probs_np[idx])

        # Default fallback: index 1
        return float(probs_np[1]) if len(probs_np) > 1 else float(probs_np[0])

    def predict(self, preprocessed_input: Any) -> float:
        """
        Executes model forward pass across audio chunks and returns aggregated score.
        """
        self.load_model()
        chunks, _ = preprocessed_input

        if not chunks:
            raise InferenceError("No audio chunks available for inference.")

        chunk_scores = []
        try:
            with torch.no_grad():
                for chunk_tensor in chunks:
                    inputs = chunk_tensor.to(self.device)
                    outputs = self._model(inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    score = self._extract_ai_probability(probs)
                    chunk_scores.append(max(0.0, min(1.0, score)))

            return float(np.mean(chunk_scores))

        except Exception as e:
            logger.error(f"Error during audio inference: {e}")
            raise InferenceError(f"Audio detection inference failed: {e}")

    def classify(self, raw_input: Any) -> DetectionResult:
        """
        Complete audio analysis pipeline with acoustic metadata and chunk breakdown.
        """
        import time
        start_time = time.perf_counter()

        chunks, metadata = self.preprocessor.preprocess(raw_input)
        self.load_model()

        chunk_scores = []
        with torch.no_grad():
            for chunk_tensor in chunks:
                inputs = chunk_tensor.to(self.device)
                outputs = self._model(inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                chunk_scores.append(round(self._extract_ai_probability(probs), 4))

        ai_score = float(np.mean(chunk_scores)) if chunk_scores else 0.0
        peak_score = float(np.max(chunk_scores)) if chunk_scores else 0.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        verdict, confidence = compute_verdict_and_confidence(ai_score, self.threshold_cfg)

        evidence: Dict[str, Any] = {
            "raw_ai_score": round(ai_score, 4),
            "human_score": round(1.0 - ai_score, 4),
            "peak_chunk_score": round(peak_score, 4),
            "duration_seconds": metadata.get("duration_seconds", 0.0),
            "original_sample_rate": metadata.get("original_sample_rate", 16000),
            "target_sample_rate": metadata.get("target_sample_rate", 16000),
            "num_chunks_analyzed": len(chunks),
            "chunk_scores": chunk_scores,
            "device": str(self.device),
        }

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
            evidence=evidence,
        )
