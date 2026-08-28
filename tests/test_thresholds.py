"""
Unit tests for 3-way classification decision logic and confidence assignment.
"""
from src.core.config import ThresholdConfig
from src.detectors.base import ConfidenceLevel, Verdict, compute_verdict_and_confidence


def test_ai_generated_verdict():
    """Scores at or above upper threshold should be LIKELY AI-GENERATED."""
    cfg = ThresholdConfig(upper_threshold=0.65, lower_threshold=0.35)
    
    verdict, conf = compute_verdict_and_confidence(0.92, cfg)
    assert verdict == Verdict.LIKELY_AI
    assert conf == ConfidenceLevel.HIGH

    verdict, conf = compute_verdict_and_confidence(0.70, cfg)
    assert verdict == Verdict.LIKELY_AI
    assert conf == ConfidenceLevel.MEDIUM


def test_human_created_verdict():
    """Scores at or below lower threshold should be LIKELY HUMAN-CREATED."""
    cfg = ThresholdConfig(upper_threshold=0.65, lower_threshold=0.35)
    
    verdict, conf = compute_verdict_and_confidence(0.08, cfg)
    assert verdict == Verdict.LIKELY_HUMAN
    assert conf == ConfidenceLevel.HIGH

    verdict, conf = compute_verdict_and_confidence(0.30, cfg)
    assert verdict == Verdict.LIKELY_HUMAN
    assert conf == ConfidenceLevel.MEDIUM


def test_uncertain_verdict():
    """Scores between lower and upper threshold should be UNCERTAIN."""
    cfg = ThresholdConfig(upper_threshold=0.65, lower_threshold=0.35)
    
    verdict, conf = compute_verdict_and_confidence(0.50, cfg)
    assert verdict == Verdict.UNCERTAIN
    assert conf == ConfidenceLevel.LOW

    verdict, conf = compute_verdict_and_confidence(0.40, cfg)
    assert verdict == Verdict.UNCERTAIN
    assert conf == ConfidenceLevel.LOW


def test_score_clamping():
    """Scores outside [0, 1] range should be clamped safely."""
    cfg = ThresholdConfig(upper_threshold=0.65, lower_threshold=0.35)
    
    verdict_high, _ = compute_verdict_and_confidence(1.5, cfg)
    assert verdict_high == Verdict.LIKELY_AI

    verdict_low, _ = compute_verdict_and_confidence(-0.5, cfg)
    assert verdict_low == Verdict.LIKELY_HUMAN
