"""
Synthetic Audio & Voice Clone Detection pipeline for Authentica AI using Wav2Vec2.

INFERENCE CONTRACT (Critical):
  All inference routes through _infer_waveform(waveform: np.ndarray) -> float.
  This function applies the loaded Hugging Face feature extractor to the raw 1-D
  float32 waveform BEFORE calling the model, matching the model card documented usage:
      audio waveform
      -> feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
      -> model(**inputs)
      -> softmax
      -> AI score extracted via verified id2label mapping

  Direct calls to self._model(raw_tensor) are PROHIBITED outside _infer_waveform.

AGGREGATION:
  All chunk score aggregation routes through _aggregate_scores(chunk_scores).
  Strategy is controlled by self.aggregation_strategy:
    "mean"         - Simple arithmetic mean across all chunks.
    "peak"         - Maximum score (highest recall for AI).
    "mean_peak"    - Conditional 60/40 mean+peak fusion when peak >= 0.65 (default).

SCORE NOMENCLATURE:
  Output is an Uncalibrated AI Detection Score — a raw softmax output, NOT a
  calibrated posterior probability.

VERDICT WORDING:
  Low scores produce "LIKELY HUMAN-LIKE". Absence of a synthetic detection signal
  does NOT establish human authorship without independent provenance evidence.

OUT-OF-DOMAIN LIMITATION:
  This model was trained on English speech from a limited set of generators
  (ElevenLabs, Amazon Polly, Kokoro, Hume AI, Speechify, Luvvoice, ASVspoof).
  Generators not in that training distribution (e.g. Google Gemini TTS, XTTS-v2,
  zero-shot multilingual TTS) are out-of-domain. Low scores on out-of-domain
  generators reflect a generalization limitation of the checkpoint, not confirmed
  human origin.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.core.config import AudioModelConfig, ThresholdConfig
from src.core.exceptions import InferenceError, ModelLoadError
from src.core.logging import get_logger
from src.detectors.base import BaseDetector, DetectionResult, compute_verdict_and_confidence
from src.preprocessing.audio import AudioPreprocessor

logger = get_logger("authentica.audio_detector")

AggregationStrategy = Literal["mean", "peak", "mean_peak"]


class AudioDetector(BaseDetector):
    """
    Wav2Vec2 Synthetic Audio & Voice Clone Detector.

    Estimates whether input speech was produced by an AI synthesis system or
    reflects the acoustic patterns of real human speech learned by the model
    during training. The model learns acoustic patterns associated with real
    and synthetic speech in its training data.

    This detector does NOT:
      - Perform cryptographic watermark verification (SynthID, C2PA).
      - Claim universal coverage of all AI audio generators.
      - Assert human authorship from a low detection score alone.
    """
    MODALITY = "audio"

    def __init__(
        self,
        model_config: Optional[AudioModelConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None,
        aggregation_strategy: AggregationStrategy = "mean_peak",
    ):
        self.config = model_config or AudioModelConfig()
        super().__init__(
            model_id=self.config.primary,
            threshold_cfg=threshold_config,
        )
        self.aggregation_strategy: AggregationStrategy = aggregation_strategy
        self.preprocessor = AudioPreprocessor(
            target_sr=self.config.target_sample_rate,
            chunk_duration_sec=self.config.chunk_duration_sec,
        )
        self.device = self._resolve_device(self.config.device)
        self._model = None
        self._feature_extractor = None
        self._id2label: Dict[int, str] = {}
        # Confirmed by feature extractor at load time.
        self._fe_sampling_rate: int = self.config.target_sample_rate

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, requested_device: str) -> torch.device:
        """Determines computation device based on user preference and hardware."""
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested_device)

    def load_model(self) -> None:
        """
        Loads the Wav2Vec2 classification model AND its feature extractor.
        Both must be present before any inference.
        """
        if self._model is not None and self._feature_extractor is not None:
            return

        try:
            logger.info(
                f"Loading audio detection model '{self.model_id}' on device '{self.device}'..."
            )
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()

            # Read the sampling rate the extractor expects.
            if hasattr(self._feature_extractor, "sampling_rate"):
                self._fe_sampling_rate = int(self._feature_extractor.sampling_rate)

            # Cache the verified id2label mapping.
            if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                self._id2label = {
                    int(k): str(v).lower()
                    for k, v in self._model.config.id2label.items()
                }
            else:
                # Fallback for garystafford checkpoint (verified: 0=real, 1=fake).
                self._id2label = {0: "real", 1: "fake"}

            logger.info(
                f"Audio model loaded. id2label={self._id2label}  "
                f"fe_sampling_rate={self._fe_sampling_rate}  device={self.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load audio model '{self.model_id}': {e}")
            raise ModelLoadError(
                f"Could not load audio detection model '{self.model_id}': {e}"
            )

    # ------------------------------------------------------------------
    # Core inference contract  (ONLY place self._model is called)
    # ------------------------------------------------------------------

    def _infer_waveform(self, waveform: np.ndarray) -> float:
        """
        Authoritative single-waveform inference function.

        Accepts a 1-D float32 waveform at self._fe_sampling_rate Hz.
        Applies the loaded Hugging Face feature extractor (which performs
        model-required normalisation per the model card), moves every returned
        tensor to self.device, calls self._model(**inputs), applies softmax,
        and extracts the uncalibrated AI detection score via id2label.

        Args:
            waveform: 1-D numpy float32 array at self._fe_sampling_rate Hz.

        Returns:
            Uncalibrated AI detection score in [0.0, 1.0].
        """
        # Feature extractor applies zero-mean / unit-variance normalisation
        # as required by Wav2Vec2 — NOT ad-hoc peak normalisation.
        inputs = self._feature_extractor(
            waveform,
            sampling_rate=self._fe_sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        # Move every tensor in the inputs dict to the target device.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = F.softmax(logits, dim=-1)

        return self._extract_ai_probability(probs)

    # ------------------------------------------------------------------
    # Score extraction and aggregation helpers
    # ------------------------------------------------------------------

    def _extract_ai_probability(self, probs: torch.Tensor) -> float:
        """
        Extracts the AI/synthetic score from a (1, num_labels) softmax tensor
        using the verified id2label mapping.

        Priority order:
          1. Any label containing 'fake', 'spoof', 'synth', 'ai', 'gen'.
          2. Invert any label containing 'real', 'bona', 'human'.
          3. Fallback to index 1 (conventional binary fake=1 assumption).
        """
        probs_np = probs.cpu().detach().numpy()[0]

        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["fake", "spoof", "synth", "ai", "gen"]):
                return float(np.clip(probs_np[idx], 0.0, 1.0))

        for idx, label_name in self._id2label.items():
            if any(term in label_name for term in ["real", "bona", "human"]):
                return float(np.clip(1.0 - probs_np[idx], 0.0, 1.0))

        return float(np.clip(probs_np[1] if len(probs_np) > 1 else probs_np[0], 0.0, 1.0))

    def _aggregate_scores(self, chunk_scores: List[float]) -> float:
        """
        Centralized aggregation used identically by predict() and classify().

        Strategies:
          "mean"      - Arithmetic mean of all chunk scores.
          "peak"      - Maximum chunk score (highest AI recall).
          "mean_peak" - If peak >= 0.65: 0.60*mean + 0.40*peak; else pure mean.

        Returns:
            Aggregated uncalibrated AI detection score in [0.0, 1.0].
        """
        if not chunk_scores:
            return 0.0

        mean_score = float(np.mean(chunk_scores))
        peak_score = float(np.max(chunk_scores))

        if self.aggregation_strategy == "mean":
            return mean_score
        elif self.aggregation_strategy == "peak":
            return peak_score
        else:  # "mean_peak" (production default)
            if peak_score >= 0.65:
                return 0.6 * mean_score + 0.4 * peak_score
            return mean_score

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def preprocess(self, raw_input: Any) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Loads audio and splits it into raw numpy waveform chunks.
        Feature extractor normalisation is applied in _infer_waveform(), not here.

        Returns:
            (chunk_waveforms, metadata) where each chunk is a 1-D float32 ndarray.
        """
        return self.preprocessor.load_and_chunk(raw_input)

    def predict(self, preprocessed_input: Any) -> float:
        """
        Runs chunked inference via _infer_waveform() and returns the aggregated
        uncalibrated AI score.  Uses the same aggregation path as classify().
        """
        self.load_model()
        chunks, _ = preprocessed_input

        if not chunks:
            raise InferenceError("No audio chunks available for inference.")

        try:
            chunk_scores = [self._infer_waveform(chunk) for chunk in chunks]
            return self._aggregate_scores(chunk_scores)
        except Exception as e:
            logger.error(f"Error during audio inference: {e}")
            raise InferenceError(f"Audio detection inference failed: {e}")

    def classify(self, raw_input: Any) -> DetectionResult:
        """
        Complete audio analysis pipeline.

        Steps:
          1. Load & chunk waveform via preprocessor.
          2. Run _infer_waveform() on each chunk (feature extractor applied here).
          3. Aggregate chunk scores via _aggregate_scores().
          4. Compute verdict and confidence.
          5. Return structured DetectionResult with full evidence dict.
        """
        import time
        start_time = time.perf_counter()

        self.load_model()
        chunks, metadata = self.preprocessor.load_and_chunk(raw_input)

        if not chunks:
            raise InferenceError("Audio preprocessing produced no chunks.")

        try:
            chunk_scores = [round(self._infer_waveform(chunk), 4) for chunk in chunks]
        except Exception as e:
            logger.error(f"Error during audio inference: {e}")
            raise InferenceError(f"Audio detection inference failed: {e}")

        ai_score = self._aggregate_scores(chunk_scores)
        mean_score = float(np.mean(chunk_scores))
        peak_score = float(np.max(chunk_scores))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        verdict, confidence = compute_verdict_and_confidence(ai_score, self.threshold_cfg)

        evidence: Dict[str, Any] = {
            "uncalibrated_ai_score": round(ai_score, 4),
            "human_like_score": round(1.0 - ai_score, 4),
            "mean_chunk_score": round(mean_score, 4),
            "peak_chunk_score": round(peak_score, 4),
            "aggregation_strategy": self.aggregation_strategy,
            "duration_seconds": metadata.get("duration_seconds", 0.0),
            "original_sample_rate": metadata.get("original_sample_rate", self._fe_sampling_rate),
            "target_sample_rate": self._fe_sampling_rate,
            "num_chunks_analyzed": len(chunks),
            "chunk_scores": chunk_scores,
            "device": str(self.device),
            "out_of_domain_warning": (
                "This acoustic model was trained on a limited set of English synthetic "
                "speech generators. Performance on unseen generators (e.g. Google Gemini "
                "TTS, XTTS-v2, multilingual zero-shot TTS) may differ from in-domain "
                "benchmarks. A low score does not confirm human authorship without "
                "independent provenance evidence."
            ),
        }

        return DetectionResult(
            modality=self.MODALITY,
            score=ai_score,
            verdict=verdict,
            confidence=confidence,
            model_name=self.model_id,
            model_version="1.0.0",
            processing_time_ms=elapsed_ms,
            disclaimer=(
                "This result is an AI-based estimate and is not definitive proof of "
                "whether the content was generated by AI. Scores are uncalibrated softmax "
                "outputs, not posterior probabilities."
            ),
            is_calibrated=False,
            evidence=evidence,
        )


