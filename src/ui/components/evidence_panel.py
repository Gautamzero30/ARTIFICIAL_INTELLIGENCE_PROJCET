"""
Authentica AI — Reusable Evidence & Multimodal Panel Component.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from src.detectors.base import DetectionResult


def render_image_evidence(result: DetectionResult, pil_img: Image.Image) -> None:
    """
    Renders image forensic evidence factors and technical execution details.
    """
    factors = result.evidence.get("forensic_factors", [])
    if factors:
        st.markdown('<div style="font-family: monospace; font-size: 0.88rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin: 1.2rem 0 0.6rem 0;">Detection Evidence</div>', unsafe_allow_html=True)
        for f in factors:
            st.markdown(f'<div style="padding: 0.5rem 0.8rem; border-radius: 6px; background: rgba(30, 41, 59, 0.7); margin-bottom: 0.35rem; border-left: 3px solid #38BDF8; font-size: 0.88rem; color: #E2E8F0;">- {f}</div>', unsafe_allow_html=True)

    with st.expander("Technical Analysis Details", expanded=False):
        freq = result.evidence.get("frequency_analysis", {})
        lap_var = freq.get("laplacian_variance", "N/A")
        hl_ratio = freq.get("high_low_frequency_ratio", "N/A")
        smooth = freq.get("texture_smoothness", "N/A")
        dims = result.evidence.get("dimensions", "N/A")
        dev = result.evidence.get("device", "cpu")
        lat = result.processing_time_ms

        st.markdown(
            f"""
            <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 0.9rem 1.1rem; font-family: monospace; font-size: 0.84rem; color: #CBD5E1; line-height: 1.7;">
                <b>Model ID:</b> {result.model_name}<br>
                <b>Architecture:</b> Vision Transformer (ViT-Base/16-224)<br>
                <b>Input Dimensions:</b> {dims}<br>
                <b>Laplacian Variance (Sharpness):</b> {lap_var}<br>
                <b>High/Low Frequency Ratio:</b> {hl_ratio}<br>
                <b>Texture Smoothness:</b> {smooth}<br>
                <b>Inference Device:</b> {dev}<br>
                <b>Inference Latency:</b> {lat:.2f} ms
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_video_evidence(result: DetectionResult) -> None:
    """
    Renders multimodal evidence (Visual Score, Audio Score, Final Fusion),
    temporal keyframe chart, and technical details.
    """
    v_sc = result.evidence.get("visual_score")
    a_sc = result.evidence.get("audio_score")
    f_sc = result.score
    mode = result.evidence.get("fusion_mode", "visual_only")
    v_w = result.evidence.get("applied_visual_weight", 0.6)
    a_w = result.evidence.get("applied_audio_weight", 0.4)

    # Multimodal Evidence Card
    has_audio = a_sc is not None
    audio_display = f"{a_sc * 100.0:.1f}%" if has_audio else "Not available — silent video"
    visual_display = f"{v_sc * 100.0:.1f}%" if v_sc is not None else "N/A"
    fusion_display = f"{f_sc * 100.0:.1f}%"

    if not has_audio:
        fallback_note = '<div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">Final score calculated from visual evidence because no audio stream was available (100% visual weight).</div>'
    else:
        fallback_note = f'<div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">Weighted late fusion: <code>S = ({v_w:.2f} &times; {visual_display}) + ({a_w:.2f} &times; {audio_display}) = {fusion_display}</code></div>'

    multi_card_html = f"""
    <div style="background: #0F172A; border: 1.5px solid #334155; border-radius: 10px; padding: 1.2rem 1.4rem; margin-top: 1rem; margin-bottom: 1.2rem;">
        <div style="font-family: monospace; font-size: 0.82rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.6rem;">Multimodal Evidence Breakdown</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8rem; text-align: center; background: rgba(30, 41, 59, 0.6); padding: 0.8rem; border-radius: 8px; border: 1px solid #334155;">
            <div>
                <div style="font-size: 0.74rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Visual Analysis</div>
                <div style="font-family: monospace; font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin-top: 0.15rem;">{visual_display}</div>
            </div>
            <div>
                <div style="font-size: 0.74rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Audio Analysis</div>
                <div style="font-family: monospace; font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin-top: 0.15rem;">{audio_display}</div>
            </div>
            <div>
                <div style="font-size: 0.74rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Final Fusion</div>
                <div style="font-family: monospace; font-size: 1.15rem; font-weight: 800; color: #38BDF8; margin-top: 0.15rem;">{fusion_display}</div>
            </div>
        </div>
        {fallback_note}
    </div>
    """
    st.markdown(multi_card_html, unsafe_allow_html=True)

    # Temporal Keyframe Profile Chart
    frames = result.evidence.get("frame_by_frame_analysis", [])
    if frames:
        st.markdown('<div style="font-family: monospace; font-size: 0.88rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; margin: 1rem 0 0.6rem 0;">Temporal Keyframe Risk Profile</div>', unsafe_allow_html=True)
        df_v = pd.DataFrame(frames)
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=[f"F{row['frame_index']} ({row['timestamp_sec']}s)" for _, row in df_v.iterrows()],
            y=[row['ai_score'] * 100.0 for _, row in df_v.iterrows()],
            mode="lines+markers",
            name="Keyframe AI Score",
            line=dict(color="#38BDF8", width=3),
            marker=dict(size=8, color=["#F43F5E" if s >= 0.45 else ("#10B981" if s < 0.40 else "#F59E0B") for s in df_v['ai_score']]),
            text=[f"{s*100.0:.1f}%" for s in df_v['ai_score']],
            textposition="top center",
        ))
        fig_v.add_hline(y=45, line_dash="dash", line_color="#F43F5E", annotation_text="AI Threshold (45%)", annotation_position="top left")
        fig_v.add_hline(y=40, line_dash="dash", line_color="#10B981", annotation_text="Human Threshold (40%)", annotation_position="bottom left")
        fig_v.update_layout(
            xaxis_title="Keyframe Timestamp",
            yaxis_title="Visual AI Score %",
            yaxis_range=[0, 105],
            height=250,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(family="sans-serif", color="#CBD5E1"),
        )
        st.plotly_chart(fig_v, use_container_width=True)

    # Technical Details Expander
    with st.expander("Technical Analysis Details", expanded=False):
        sub_models = result.evidence.get("sub_models", {})
        v_w_pct = int(round(v_w * 100.0))
        a_w_pct = int(round(a_w * 100.0))
        mode_str = "Dual-Stream Weighted Late Fusion" if mode == "dual_stream" else "Visual-Only Single Modality"
        vis_mod = sub_models.get("visual_model", "umm-maybe/AI-image-detector")
        aud_mod = sub_models.get("audio_model", "N/A")
        proc_sec = result.processing_time_ms / 1000.0
        n_frames = len(frames)

        st.markdown(
            f"""
            <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 0.9rem 1.1rem; font-family: monospace; font-size: 0.84rem; color: #CBD5E1; line-height: 1.7;">
                <b>Fusion Strategy:</b> {mode.upper()} ({mode_str})<br>
                <b>Applied Visual Weight:</b> {v_w_pct}%<br>
                <b>Applied Audio Weight:</b> {a_w_pct}%<br>
                <b>Evaluated Keyframes:</b> {n_frames} frames uniformly sampled<br>
                <b>Visual Model Backbone:</b> {vis_mod}<br>
                <b>Audio Model Backbone:</b> {aud_mod}<br>
                <b>Total Processing Latency:</b> {proc_sec:.2f}s
            </div>
            """,
            unsafe_allow_html=True,
        )
