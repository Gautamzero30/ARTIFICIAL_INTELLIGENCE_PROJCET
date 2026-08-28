"""
Multimodal Late Fusion Engine for combining visual and acoustic detection scores.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.core.exceptions import ProcessingError, ValidationError


@dataclass(frozen=True)
class FusionResult:
    fused_score: float
    visual_score: Optional[float]
    audio_score: Optional[float]
    visual_weight: float
    audio_weight: float
    fusion_mode: str  # "dual_stream", "visual_only", "audio_only"
    evidence: Dict[str, Any]


class LateFusionEngine:
    """
    Combines visual keyframe scores and acoustic waveform scores using weighted late fusion.
    Handles silent videos and single-modality fallback paths gracefully.
    """

    def __init__(
        self,
        visual_weight: float = 0.6,
        audio_weight: float = 0.4,
    ):
        if abs((visual_weight + audio_weight) - 1.0) > 1e-4:
            raise ValidationError(
                f"Fusion weights must sum to 1.0. Got visual_weight={visual_weight}, audio_weight={audio_weight}"
            )
        self.default_visual_weight = visual_weight
        self.default_audio_weight = audio_weight

    def fuse(
        self,
        visual_score: Optional[float],
        audio_score: Optional[float],
        visual_evidence: Optional[Dict[str, Any]] = None,
        audio_evidence: Optional[Dict[str, Any]] = None,
    ) -> FusionResult:
        """
        Executes late fusion calculation.
        """
        v_ev = visual_evidence or {}
        a_ev = audio_evidence or {}

        # Case 1: Both visual and audio scores available (Dual-stream fusion)
        if visual_score is not None and audio_score is not None:
            v_score = max(0.0, min(1.0, float(visual_score)))
            a_score = max(0.0, min(1.0, float(audio_score)))

            fused = (self.default_visual_weight * v_score) + (self.default_audio_weight * a_score)
            fused_clamped = max(0.0, min(1.0, float(fused)))

            evidence = {
                "visual_score": round(v_score, 4),
                "audio_score": round(a_score, 4),
                "applied_visual_weight": self.default_visual_weight,
                "applied_audio_weight": self.default_audio_weight,
                "visual_evidence": v_ev,
                "audio_evidence": a_ev,
            }

            return FusionResult(
                fused_score=fused_clamped,
                visual_score=v_score,
                audio_score=a_score,
                visual_weight=self.default_visual_weight,
                audio_weight=self.default_audio_weight,
                fusion_mode="dual_stream",
                evidence=evidence,
            )

        # Case 2: Visual only (e.g. silent video)
        elif visual_score is not None and audio_score is None:
            v_score = max(0.0, min(1.0, float(visual_score)))
            evidence = {
                "visual_score": round(v_score, 4),
                "audio_score": None,
                "note": "Video contains no audio stream or audio analysis was unavailable. Visual score applied at 100% weight.",
                "visual_evidence": v_ev,
            }
            return FusionResult(
                fused_score=v_score,
                visual_score=v_score,
                audio_score=None,
                visual_weight=1.0,
                audio_weight=0.0,
                fusion_mode="visual_only",
                evidence=evidence,
            )

        # Case 3: Audio only (e.g. corrupt visual frames)
        elif visual_score is None and audio_score is not None:
            a_score = max(0.0, min(1.0, float(audio_score)))
            evidence = {
                "visual_score": None,
                "audio_score": round(a_score, 4),
                "note": "Visual stream was unavailable. Audio score applied at 100% weight.",
                "audio_evidence": a_ev,
            }
            return FusionResult(
                fused_score=a_score,
                visual_score=None,
                audio_score=a_score,
                visual_weight=0.0,
                audio_weight=1.0,
                fusion_mode="audio_only",
                evidence=evidence,
            )

        # Case 4: Both failed
        else:
            raise ProcessingError("Both visual and audio analyses failed. Unable to compute multimodal video fusion.")
