"""
Text Detection pipeline for Authentica AI using RoBERTa transformers.
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.core.config import TextModelConfig, ThresholdConfig
from src.core.exceptions import InferenceError, ModelLoadError
from src.core.logging import get_logger
from src.detectors.base import BaseDetector, DetectionResult, compute_verdict_and_confidence
from src.preprocessing.text import TextPreprocessor

logger = get_logger("authentica.text_detector")


class TextDetector(BaseDetector):
    """
    RoBERTa-based AI-Text Detector.
    Estimates whether input text is AI-generated (e.g. ChatGPT, GPT-4, Claude)
    or human-written.
    """
    MODALITY = "text"

    def __init__(
        self,
        model_config: Optional[TextModelConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None,
    ):
        self.config = model_config or TextModelConfig()
        super().__init__(
            model_id=self.config.primary,
            threshold_cfg=threshold_config,
        )
        self.preprocessor = TextPreprocessor(
            min_characters=self.config.min_character_length,
            max_tokens=self.config.max_length,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.device = self._resolve_device(self.config.device)
        self._model = None
        self._tokenizer = None
        self._id2label = {}

    def _resolve_device(self, requested_device: str) -> torch.device:
        """Determines computation device based on user preference and hardware."""
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested_device)

    def load_model(self) -> None:
        """
        Loads the RoBERTa classification model and tokenizer from Hugging Face.
        """
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            logger.info(f"Loading text detection model '{self.model_id}' on device '{self.device}'...")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()

            # Cache label mapping
            if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                self._id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
            else:
                # Default binary assumption: 0=Human, 1=ChatGPT/AI
                self._id2label = {0: "human", 1: "chatgpt"}

            logger.info(f"Text detection model loaded successfully. Label map: {self._id2label}")

        except Exception as e:
            logger.error(f"Failed to load text model '{self.model_id}': {e}")
            raise ModelLoadError(f"Could not load text detection model '{self.model_id}': {e}")

    def preprocess(self, raw_input: Any) -> List[Dict[str, torch.Tensor]]:
        """
        Validates text and converts into tokenized chunk tensors.
        """
        self.load_model()
        return self.preprocessor.chunk_tokens(raw_input, self._tokenizer)

    def _extract_ai_probability(self, probs: torch.Tensor) -> float:
        """
        Extracts AI probability score S_AI from output softmax probabilities.
        """
        probs_np = probs.cpu().detach().numpy()[0]

        # 1. Look for explicit AI / ChatGPT / Fake label index
        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["chatgpt", "ai", "fake", "synth", "gen"]):
                return float(probs_np[idx])

        # 2. Look for human / real label and invert
        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["human", "real"]):
                return float(1.0 - probs_np[idx])

        # Default fallback: index 1
        return float(probs_np[1]) if len(probs_np) > 1 else float(probs_np[0])

    def predict(self, preprocessed_chunks: List[Dict[str, torch.Tensor]]) -> float:
        """
        Executes forward passes across all chunks and aggregates into an overall AI score.
        """
        self.load_model()

        if not preprocessed_chunks:
            raise InferenceError("No token chunks available for inference.")

        chunk_scores = []
        try:
            with torch.no_grad():
                for chunk in preprocessed_chunks:
                    inputs = {k: v.to(self.device) for k, v in chunk.items()}
                    outputs = self._model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    score = self._extract_ai_probability(probs)
                    chunk_scores.append(max(0.0, min(1.0, score)))

            # Aggregate chunk scores (mean score)
            aggregated_score = float(np.mean(chunk_scores))
            return aggregated_score

        except Exception as e:
            logger.error(f"Error during text inference: {e}")
            raise InferenceError(f"Text detection inference failed: {e}")

    def classify(self, raw_input: Any) -> DetectionResult:
        """
        Complete text analysis pipeline with chunk breakdown and linguistic evidence.
        """
        import time
        start_time = time.perf_counter()

        cleaned_text = self.preprocessor.validate_text(str(raw_input))
        word_count = len(cleaned_text.split())
        char_count = len(cleaned_text)
        is_non_english = self.preprocessor.detect_non_english(cleaned_text)

        chunks = self.preprocess(cleaned_text)
        
        # Run inference per chunk to collect individual chunk scores
        chunk_scores = []
        with torch.no_grad():
            for chunk in chunks:
                inputs = {k: v.to(self.device) for k, v in chunk.items()}
                outputs = self._model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                chunk_scores.append(round(self._extract_ai_probability(probs), 4))

        ai_score = float(np.mean(chunk_scores)) if chunk_scores else 0.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        verdict, confidence = compute_verdict_and_confidence(ai_score, self.threshold_cfg)

        evidence: Dict[str, Any] = {
            "raw_ai_score": round(ai_score, 4),
            "human_score": round(1.0 - ai_score, 4),
            "word_count": word_count,
            "character_count": char_count,
            "num_chunks_analyzed": len(chunks),
            "chunk_scores": chunk_scores,
            "device": str(self.device),
        }

        if is_non_english:
            evidence["language_warning"] = (
                "Text appears to contain non-Latin/multilingual script. "
                "The primary model is optimized for English, so results should be treated with higher uncertainty."
            )

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
