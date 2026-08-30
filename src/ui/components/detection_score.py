"""
Authentica AI — Reusable Segmented Forensic Decision Scale Component.
"""
import streamlit as st
from src.detectors.base import Verdict


def render_detection_score(
    score: float,
    lower_threshold: float = 0.40,
    upper_threshold: float = 0.45,
    verdict: Verdict = Verdict.UNCERTAIN,
) -> None:
    score_pct = max(0.0, min(100.0, float(score) * 100.0))
    low_pct = float(lower_threshold) * 100.0
    high_pct = float(upper_threshold) * 100.0

    if score >= upper_threshold:
        theme_color = "#F43F5E"
        region_desc = "AI-GENERATED REGION (Score &ge; 45%)"
    elif score < lower_threshold:
        theme_color = "#10B981"
        region_desc = "HUMAN-CREATED REGION (Score &lt; 40%)"
    else:
        theme_color = "#F59E0B"
        region_desc = "UNCERTAINTY BAND (40% &le; Score &lt; 45%)"

    score_card_html = f"""<div style="background: #0F172A; border: 1.5px solid #334155; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
<div>
<span style="font-family: monospace; font-size: 0.8rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase;">Forensic Decision Scale</span>
<div style="font-size: 0.82rem; color: #64748B; margin-top: 0.15rem;">Uncalibrated AI Likeness Metric</div>
</div>
<div style="text-align: right;">
<span style="font-family: monospace; font-size: 1.6rem; font-weight: 800; color: {theme_color};">{score_pct:.1f}%</span>
</div>
</div>
<div style="position: relative; width: 100%; height: 20px; background: #1E293B; border-radius: 10px; margin: 1.6rem 0 2rem 0; border: 1px solid #334155;">
<div style="position: absolute; left: 0%; width: {low_pct}%; height: 100%; background: rgba(16, 185, 129, 0.25); border-top-left-radius: 9px; border-bottom-left-radius: 9px; border-right: 2px solid #10B981;">
<span style="position: absolute; top: -20px; left: 8px; font-family: monospace; font-size: 0.72rem; color: #10B981; font-weight: 700;">HUMAN (&lt;40%)</span>
</div>
<div style="position: absolute; left: {low_pct}%; width: {high_pct - low_pct}%; height: 100%; background: rgba(245, 158, 11, 0.55); border-right: 2px solid #F43F5E;">
<span style="position: absolute; top: -20px; left: -10px; font-family: monospace; font-size: 0.70rem; color: #F59E0B; font-weight: 700; white-space: nowrap;">40–45%</span>
</div>
<div style="position: absolute; left: {high_pct}%; width: {100.0 - high_pct}%; height: 100%; background: rgba(244, 63, 94, 0.25); border-top-right-radius: 9px; border-bottom-right-radius: 9px;">
<span style="position: absolute; top: -20px; right: 8px; font-family: monospace; font-size: 0.72rem; color: #F43F5E; font-weight: 700;">AI (&ge;45%)</span>
</div>
<div style="position: absolute; left: {score_pct}%; top: -8px; width: 16px; height: 36px; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; z-index: 10;">
<div style="width: 16px; height: 16px; border-radius: 50%; background: #FFFFFF; border: 3.5px solid {theme_color}; box-shadow: 0 0 10px rgba(0,0,0,0.8);"></div>
<div style="width: 2px; height: 16px; background: {theme_color};"></div>
</div>
</div>
<div style="display: flex; justify-content: space-between; font-family: monospace; font-size: 0.75rem; color: #64748B; margin-top: -1rem;">
<span>0.0%</span>
<span style="color: #94A3B8;">Position: {region_desc}</span>
<span>100.0%</span>
</div>
<div style="margin-top: 1rem; padding: 0.65rem 0.9rem; background: rgba(15, 23, 42, 0.6); border-radius: 6px; border-left: 3px solid {theme_color}; font-size: 0.84rem; color: #CBD5E1; line-height: 1.5;">
<strong>Note:</strong> The AI Detection Score ({score_pct:.1f}%) represents model-estimated AI-likeness, not a physical ratio of AI vs human content.
</div>
</div>"""
    st.markdown(score_card_html, unsafe_allow_html=True)
