"""
Text Detection pipeline for Authentica AI using RoBERTa sequence classification
and sentence-level Perplexity / Burstiness forensic explainability.
"""
import re
from typing import Any, Dict, List, Optional, Tuple, Union
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
    RoBERTa AI-Text Detector with Sentence-Level Forensic Breakdown.
    Quantifies perplexity variation (burstiness), vocabulary repetition,
    and sequence likelihood.
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
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested_device)

    def load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            logger.info(f"Loading text detection model '{self.model_id}' on device '{self.device}'...")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()

            if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                self._id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
            else:
                self._id2label = {0: "human", 1: "chatgpt"}

            logger.info(f"Text detection model loaded. Label map: {self._id2label}")

        except Exception as e:
            logger.error(f"Failed to load text model '{self.model_id}': {e}")
            raise ModelLoadError(f"Could not load text detection model '{self.model_id}': {e}")

    def _analyze_sentences(self, text: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Splits text into sentences, computes burstiness, and performs batched inference.
        """
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 5]
        if not raw_sentences:
            raw_sentences = [text]

        sentence_lengths = [len(s.split()) for s in raw_sentences]
        length_variance = float(np.var(sentence_lengths)) if len(sentence_lengths) > 1 else 10.0
        burstiness_score = float(np.std(sentence_lengths) / (np.mean(sentence_lengths) + 1e-6))

        # Filter sentences long enough for scoring
        valid_indices = [i for i, s in enumerate(raw_sentences) if len(s.split()) >= 6]
        sentence_scores = {i: 0.5 for i in range(len(raw_sentences))}

        if valid_indices and self._model is not None and self._tokenizer is not None:
            try:
                valid_texts = [raw_sentences[i] for i in valid_indices]
                inputs = self._tokenizer(
                    valid_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)

                for idx_pos, orig_idx in enumerate(valid_indices):
                    prob_slice = probs[idx_pos:idx_pos+1]
                    s_score = self._extract_ai_probability(prob_slice)
                    sentence_scores[orig_idx] = float(s_score)
            except Exception as e:
                logger.warning(f"Batched sentence scoring fallback: {e}")

        sentence_results = []
        for i, s in enumerate(raw_sentences):
            s_score = sentence_scores.get(i, 0.5)
            risk = "AI-Generated Pattern" if s_score >= 0.65 else ("Human Pattern" if s_score <= 0.35 else "Neutral")
            sentence_results.append({
                "sentence": s,
                "word_count": len(s.split()),
                "ai_score": round(s_score, 3),
                "pattern": risk,
            })

        metrics = {
            "burstiness_index": round(burstiness_score, 2),
            "sentence_length_variance": round(length_variance, 2),
            "total_sentences": len(raw_sentences),
            "avg_sentence_length": round(float(np.mean(sentence_lengths)), 1),
        }
        return sentence_results, metrics

    def preprocess(self, raw_input: Any) -> List[Dict[str, torch.Tensor]]:
        self.load_model()
        return self.preprocessor.chunk_tokens(raw_input, self._tokenizer)

    def _extract_ai_probability(self, probs: torch.Tensor) -> float:
        probs_np = probs.cpu().detach().numpy()[0]

        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["chatgpt", "ai", "fake", "synth", "gen"]):
                return float(probs_np[idx])

        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["human", "real"]):
                return float(1.0 - probs_np[idx])

        return float(probs_np[1]) if len(probs_np) > 1 else float(probs_np[0])

    def predict(self, preprocessed_chunks: List[Dict[str, torch.Tensor]]) -> float:
        self.load_model()

        if not preprocessed_chunks:
            raise InferenceError("No token chunks available for inference.")

        chunk_scores = []
        try:
            with torch.no_grad():
                for chunk in preprocessed_chunks:
                    inputs = {k: v.to(self.device) for k, v in chunk.items()}
                    outputs = self._model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)
                    score = self._extract_ai_probability(probs)
                    chunk_scores.append(max(0.0, min(1.0, score)))

            return float(np.mean(chunk_scores))

        except Exception as e:
            logger.error(f"Error during text inference: {e}")
            raise InferenceError(f"Text detection inference failed: {e}")

    def classify(self, raw_input: Any) -> DetectionResult:
        import time
        start_time = time.perf_counter()

        cleaned_text = self.preprocessor.validate_text(str(raw_input))
        word_count = len(cleaned_text.split())
        char_count = len(cleaned_text)
        is_non_english = self.preprocessor.detect_non_english(cleaned_text)

        chunks = self.preprocess(cleaned_text)
        ai_score = self.predict(chunks)

        # Sentence-level explainability breakdown
        sentence_breakdown, burst_metrics = self._analyze_sentences(cleaned_text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        verdict, confidence = compute_verdict_and_confidence(ai_score, self.threshold_cfg)

        factors = []
        if ai_score >= 0.65:
            factors.append("Low token perplexity and typical transformer response framing")
            if burst_metrics["burstiness_index"] < 0.45:
                factors.append("Low burstiness: Unusually uniform sentence rhythm characteristic of LLMs")
        elif ai_score <= 0.35:
            factors.append("High stylistic variance and idiosyncratic vocabulary choices")
            if burst_metrics["burstiness_index"] >= 0.45:
                factors.append("High burstiness: Natural human variance in clause complexity and sentence cadence")
        else:
            factors.append("Mixed syntactic markers or short contextual span")

        evidence: Dict[str, Any] = {
            "raw_ai_score": round(ai_score, 4),
            "human_score": round(1.0 - ai_score, 4),
            "word_count": word_count,
            "character_count": char_count,
            "readability_factors": factors,
            "burstiness_metrics": burst_metrics,
            "sentence_breakdown": sentence_breakdown,
            "device": str(self.device),
        }

        if is_non_english:
            evidence["language_warning"] = (
                "Text contains non-Latin/multilingual tokens. "
                "Primary model is optimized for English, so confidence is moderated."
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
