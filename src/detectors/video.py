"""
Multimodal Video Detection orchestrator for Authentica AI.
Reuses Vision Transformer (ImageDetector) and Wav2Vec2 (AudioDetector) via weighted late fusion.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.config import ThresholdConfig, VideoModelConfig
from src.core.exceptions import ProcessingError
from src.core.logging import get_logger
from src.detectors.audio import AudioDetector
from src.detectors.base import BaseDetector, DetectionResult, compute_verdict_and_confidence
from src.detectors.image import ImageDetector
from src.fusion.late_fusion import FusionResult, LateFusionEngine
from src.preprocessing.video import VideoPreprocessor

logger = get_logger("authentica.video_detector")


class VideoDetector(BaseDetector):
    """
    Multimodal Video AI Detector combining visual keyframe analysis and acoustic deepfake detection.
    """
    MODALITY = "video"

    def __init__(
        self,
        video_config: Optional[VideoModelConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None,
        image_detector: Optional[ImageDetector] = None,
        audio_detector: Optional[AudioDetector] = None,
    ):
        self.config = video_config or VideoModelConfig()
        super().__init__(
            model_id="Authentica-Multimodal-Late-Fusion",
            threshold_cfg=threshold_config,
        )
        self.preprocessor = VideoPreprocessor(
            sample_frames=self.config.sample_frames,
            max_duration_sec=self.config.max_duration_sec,
        )
        self.fusion_engine = LateFusionEngine(
            visual_weight=self.config.visual_weight,
            audio_weight=self.config.audio_weight,
        )
        self.image_detector = image_detector or ImageDetector()
        self.audio_detector = audio_detector or AudioDetector()

    def load_model(self) -> None:
        """Loads both underlying sub-detectors."""
        self.image_detector.load_model()
        self.audio_detector.load_model()

    def preprocess(self, raw_input: Any) -> Tuple[List[Any], Optional[bytes], Dict[str, Any]]:
        """
        Extracts keyframes, audio stream, and video container metadata.
        """
        keyframes, video_meta = self.preprocessor.extract_keyframes(raw_input)
        audio_bytes, has_audio = self.preprocessor.extract_audio_track(raw_input)
        video_meta["has_audio_track"] = has_audio
        return keyframes, audio_bytes, video_meta

    def predict(self, preprocessed_input: Any) -> float:
        """
        Executes multimodal late fusion prediction.
        """
        keyframes, audio_bytes, _ = preprocessed_input

        # 1. Visual stream analysis
        frame_scores = []
        for frame in keyframes:
            res = self.image_detector.classify(frame)
            frame_scores.append(res.score)

        visual_score = float(np.mean(frame_scores)) if frame_scores else None

        # 2. Audio stream analysis
        audio_score = None
        if audio_bytes is not None:
            try:
                a_res = self.audio_detector.classify(audio_bytes)
                audio_score = a_res.score
            except Exception as e:
                logger.warning(f"Audio analysis failed during video processing: {e}")

        # 3. Multimodal late fusion
        fusion_res: FusionResult = self.fusion_engine.fuse(
            visual_score=visual_score,
            audio_score=audio_score,
        )
        return fusion_res.fused_score

    def classify(self, raw_input: Any) -> DetectionResult:
        """
        Complete video analysis pipeline with frame breakdown and late fusion evidence.
        """
        import time
        start_time = time.perf_counter()

        keyframes, audio_bytes, video_meta = self.preprocess(raw_input)

        # 1. Visual Stream Analysis across sampled keyframes
        frame_results = []
        frame_scores = []
        timestamps = video_meta.get("sample_timestamps", [])

        for idx, frame in enumerate(keyframes):
            res = self.image_detector.classify(frame)
            t_stamp = timestamps[idx] if idx < len(timestamps) else round(idx * 0.5, 2)
            frame_scores.append(res.score)
            frame_results.append({
                "frame_index": idx + 1,
                "timestamp_sec": t_stamp,
                "ai_score": round(res.score, 4),
                "verdict": res.verdict.value,
            })

        visual_score = float(np.mean(frame_scores)) if frame_scores else None
        peak_visual_score = float(np.max(frame_scores)) if frame_scores else 0.0
        peak_frame_idx = int(np.argmax(frame_scores)) if frame_scores else 0
        peak_timestamp = timestamps[peak_frame_idx] if peak_frame_idx < len(timestamps) else 0.0

        # 2. Audio Stream Analysis
        audio_score = None
        audio_evidence = {}
        if audio_bytes is not None:
            try:
                a_res = self.audio_detector.classify(audio_bytes)
                audio_score = a_res.score
                audio_evidence = a_res.evidence
            except Exception as e:
                logger.warning(f"Audio extraction succeeded but classification failed: {e}")

        # 3. Multimodal Late Fusion
        fusion_result: FusionResult = self.fusion_engine.fuse(
            visual_score=visual_score,
            audio_score=audio_score,
            visual_evidence={
                "mean_visual_score": round(visual_score, 4) if visual_score is not None else None,
                "peak_visual_score": round(peak_visual_score, 4),
                "peak_frame_timestamp_sec": peak_timestamp,
            },
            audio_evidence=audio_evidence,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        verdict, confidence = compute_verdict_and_confidence(fusion_result.fused_score, self.threshold_cfg)

        evidence: Dict[str, Any] = {
            "fused_score": round(fusion_result.fused_score, 4),
            "visual_score": round(visual_score, 4) if visual_score is not None else None,
            "audio_score": round(audio_score, 4) if audio_score is not None else None,
            "applied_visual_weight": fusion_result.visual_weight,
            "applied_audio_weight": fusion_result.audio_weight,
            "fusion_mode": fusion_result.fusion_mode,
            "video_metadata": video_meta,
            "peak_frame_anomaly": {
                "peak_score": round(peak_visual_score, 4),
                "timestamp_sec": peak_timestamp,
            },
            "frame_by_frame_analysis": frame_results,
            "sub_models": {
                "visual_model": self.image_detector.model_id,
                "audio_model": self.audio_detector.model_id if audio_score is not None else "N/A (Silent Video)",
            },
        }

        return DetectionResult(
            modality=self.MODALITY,
            score=fusion_result.fused_score,
            verdict=verdict,
            confidence=confidence,
            model_name=self.model_id,
            model_version="1.0.0",
            processing_time_ms=elapsed_ms,
            disclaimer="This result is an AI-based estimate and is not definitive proof of whether the content was generated by AI.",
            is_calibrated=False,
            evidence=evidence,
        )
