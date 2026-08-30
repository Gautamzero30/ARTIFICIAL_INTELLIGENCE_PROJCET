"""
Authentica AI — Clean Result Card Component.
"""
import textwrap
import streamlit as st
from src.detectors.base import DetectionResult, Verdict


def render_result_card(result: DetectionResult) -> None:
    score_pct = result.score * 100.0

    if result.verdict == Verdict.LIKELY_AI:
        badge_bg = "rgba(244, 63, 94, 0.15)"
        badge_border = "rgba(244, 63, 94, 0.4)"
        badge_color = "#F43F5E"
        icon_symbol = "●"
        verdict_title = "LIKELY AI-GENERATED"
        explanation = "The detector identified characteristics associated with AI-generated content. The score exceeds the configured AI decision threshold (&ge; 45%)."
    elif result.verdict == Verdict.LIKELY_HUMAN:
        badge_bg = "rgba(16, 185, 129, 0.15)"
        badge_border = "rgba(16, 185, 129, 0.4)"
        badge_color = "#10B981"
        icon_symbol = "●"
        verdict_title = "LIKELY HUMAN-CREATED"
        explanation = "The detector found relatively little evidence associated with AI-generated content. The result is classified as likely human-created under the current decision policy (&lt; 40%)."
    else:
        badge_bg = "rgba(245, 158, 11, 0.15)"
        badge_border = "rgba(245, 158, 11, 0.4)"
        badge_color = "#F59E0B"
        icon_symbol = "●"
        verdict_title = "UNCERTAIN"
        explanation = "The detection score falls within the configured uncertainty band (40%–45%). The available evidence is insufficient for a confident AI-generated or human-created classification."

    strength = result.confidence.value
    processing_sec = result.processing_time_ms / 1000.0

    card_html = f"""<div style="background: #0F172A; border: 1.5px solid #334155; border-radius: 12px; padding: 1.8rem; text-align: center; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35); margin-bottom: 1.2rem;">
<div style="display: inline-flex; align-items: center; gap: 0.5rem; background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 9999px; padding: 0.35rem 0.95rem; margin-bottom: 0.9rem;">
<span style="color: {badge_color}; font-size: 0.75rem;">{icon_symbol}</span>
<span style="font-family: monospace; font-size: 0.88rem; font-weight: 800; color: {badge_color}; letter-spacing: 0.04em;">{verdict_title}</span>
</div>
<div style="font-family: monospace; font-size: 0.8rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em;">AI Detection Score</div>
<div style="font-family: monospace; font-size: 3.2rem; font-weight: 800; color: {badge_color}; margin: 0.2rem 0 0.6rem 0;">{score_pct:.1f}%</div>
<div style="display: flex; justify-content: center; gap: 2rem; border-top: 1px solid #1E293B; padding-top: 0.8rem; margin-top: 0.4rem;">
<div>
<div style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Confidence</div>
<div style="font-family: monospace; font-size: 0.92rem; font-weight: 700; color: #F8FAFC; margin-top: 0.15rem;">{strength}</div>
</div>
<div style="border-left: 1px solid #334155; height: 28px; margin-top: 4px;"></div>
<div>
<div style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Processing</div>
<div style="font-family: monospace; font-size: 0.92rem; font-weight: 700; color: #F8FAFC; margin-top: 0.15rem;">{processing_sec:.2f}s</div>
</div>
</div>
<div style="margin-top: 1.1rem; padding: 0.7rem 0.9rem; background: rgba(30, 41, 59, 0.5); border-radius: 8px; font-size: 0.85rem; color: #CBD5E1; line-height: 1.5; text-align: left;">
{explanation}
</div>
</div>"""
    st.markdown(card_html, unsafe_allow_html=True)
