"""
Unit tests for 3-way classification decision logic and confidence assignment.
Verifies the centralized 0.40/0.45 decision policy across all modalities.
"""
import pytest
from src.core.config import ThresholdConfig
from src.detectors.base import ConfidenceLevel, Verdict, compute_verdict_and_confidence, get_verdict


def test_mandatory_decision_cases():
    """
    Required cases:
      0.00  -> LIKELY HUMAN-CREATED
      0.20  -> LIKELY HUMAN-CREATED
      0.399 -> LIKELY HUMAN-CREATED
      0.40  -> UNCERTAIN
      0.425 -> UNCERTAIN
      0.449 -> UNCERTAIN
      0.45  -> LIKELY AI-GENERATED
      0.61  -> LIKELY AI-GENERATED
      1.00  -> LIKELY AI-GENERATED
    """
    cfg = ThresholdConfig(upper_threshold=0.45, lower_threshold=0.40)

    assert get_verdict(0.00, cfg) == Verdict.LIKELY_HUMAN
    assert get_verdict(0.20, cfg) == Verdict.LIKELY_HUMAN
    assert get_verdict(0.399, cfg) == Verdict.LIKELY_HUMAN
    assert get_verdict(0.40, cfg) == Verdict.UNCERTAIN
    assert get_verdict(0.425, cfg) == Verdict.UNCERTAIN
    assert get_verdict(0.449, cfg) == Verdict.UNCERTAIN
    assert get_verdict(0.45, cfg) == Verdict.LIKELY_AI
    assert get_verdict(0.61, cfg) == Verdict.LIKELY_AI
    assert get_verdict(1.00, cfg) == Verdict.LIKELY_AI


def test_video_analysis_score_verdicts():
    """
    Specific video test cases:
      3.8%  (0.038) -> LIKELY HUMAN-CREATED
      42.0% (0.420) -> UNCERTAIN
      61.1% (0.611) -> LIKELY AI-GENERATED
      89.0% (0.890) -> LIKELY AI-GENERATED
    """
    cfg = ThresholdConfig(upper_threshold=0.45, lower_threshold=0.40)

    # 1. 3.8%
    v, _ = compute_verdict_and_confidence(0.038, cfg)
    assert v == Verdict.LIKELY_HUMAN

    # 2. 42%
    v, _ = compute_verdict_and_confidence(0.42, cfg)
    assert v == Verdict.UNCERTAIN

    # 3. 61.1% (Critical user bug case)
    v, _ = compute_verdict_and_confidence(0.611, cfg)
    assert v == Verdict.LIKELY_AI

    # 4. 89%
    v, _ = compute_verdict_and_confidence(0.89, cfg)
    assert v == Verdict.LIKELY_AI


def test_default_config_matches_policy():
    """Default ThresholdConfig must default to 0.40 and 0.45."""
    cfg = ThresholdConfig()
    assert cfg.lower_threshold == 0.40
    assert cfg.upper_threshold == 0.45


def test_score_clamping():
    """Scores outside [0, 1] range should be clamped safely."""
    cfg = ThresholdConfig(upper_threshold=0.45, lower_threshold=0.40)

    verdict_high, _ = compute_verdict_and_confidence(1.5, cfg)
    assert verdict_high == Verdict.LIKELY_AI

    verdict_low, _ = compute_verdict_and_confidence(-0.5, cfg)
    assert verdict_low == Verdict.LIKELY_HUMAN


