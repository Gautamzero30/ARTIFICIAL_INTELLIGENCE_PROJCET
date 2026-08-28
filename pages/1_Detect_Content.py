"""
Authentica AI — Enterprise Multimodal Forensic Detection Studio.
"""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.config import ThresholdConfig
from src.core.exceptions import (
    CorruptedFileError,
    FileSecurityError,
    InsufficientContentError,
    UnsupportedFormatError,
    ValidationError,
)
from src.detectors.audio import AudioDetector
from src.detectors.base import DetectionResult, Verdict
from src.detectors.image import ImageDetector
from src.detectors.text import TextDetector
from src.detectors.video import VideoDetector
from src.utils.file_validator import FileValidator

st.set_page_config(
    page_title="Forensic Detection Studio — Authentica AI",
    page_icon="🔍",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .studio-header {
        margin-bottom: 1.5rem;
    }
    .verdict-hero-ai {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .verdict-hero-human {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.05) 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .verdict-hero-uncertain {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.05) 100%);
        border: 2px solid #F59E0B;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .verdict-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .factor-tag {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.88rem;
        color: #E2E8F0;
        display: flex;
        align-items: center;
    }
    .sentence-box {
        padding: 0.6rem 0.9rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        font-size: 0.93rem;
        line-height: 1.5;
    }
    .sentence-ai {
        background-color: rgba(239, 68, 68, 0.12);
        border-left: 3px solid #EF4444;
    }
    .sentence-human {
        background-color: rgba(16, 185, 129, 0.12);
        border-left: 3px solid #10B981;
    }
    .sentence-neutral {
        background-color: rgba(148, 163, 184, 0.08);
        border-left: 3px solid #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_gauge(score: float, title: str = "AI Probability Score") -> go.Figure:
    """Renders an enterprise radial gauge meter via Plotly."""
    pct = score * 100
    color = "#EF4444" if score >= 0.65 else ("#10B981" if score <= 0.35 else "#F59E0B")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36, "weight": "bold", "color": color}},
            title={"text": title, "font": {"size": 16, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 3},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=15, r=15, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans"},
    )
    return fig


# Cached Model Singletons
@st.cache_resource(show_spinner=False)
def get_cached_image_detector() -> ImageDetector:
    det = ImageDetector()
    det.load_model()
    return det


@st.cache_resource(show_spinner=False)
def get_cached_text_detector() -> TextDetector:
    det = TextDetector()
    det.load_model()
    return det


@st.cache_resource(show_spinner=False)
def get_cached_audio_detector() -> AudioDetector:
    det = AudioDetector()
    det.load_model()
    return det


@st.cache_resource(show_spinner=False)
def get_cached_video_detector() -> VideoDetector:
    det = VideoDetector()
    det.load_model()
    return det


st.markdown('<div class="studio-header"><h1>🔍 Forensic Detection Studio</h1><p style="color: #94A3B8;">Multi-engine content verification and explainability console</p></div>', unsafe_allow_html=True)

# Sensitivity Controls in Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Detection Sensitivity")
    sensitivity = st.selectbox(
        "Operating Sensitivity Preset",
        ["Balanced (Standard 35% / 65%)", "Strict (High Recall for AI 25% / 55%)", "Permissive (High Precision 45% / 75%)"],
    )
    if "Strict" in sensitivity:
        thresh_cfg = ThresholdConfig(upper_threshold=0.55, lower_threshold=0.25)
    elif "Permissive" in sensitivity:
        thresh_cfg = ThresholdConfig(upper_threshold=0.75, lower_threshold=0.45)
    else:
        thresh_cfg = ThresholdConfig(upper_threshold=0.65, lower_threshold=0.35)

    st.caption(f"**Upper Threshold (AI):** `{thresh_cfg.upper_threshold*100:.0f}%` | **Lower (Human):** `{thresh_cfg.lower_threshold*100:.0f}%`")
    st.markdown("---")
    st.markdown("### 🔒 Security Protocols")
    st.caption("✅ Magic byte header verification enabled\n✅ EXIF auto-transposition active\n✅ Zero execution sandbox")

tab_image, tab_text, tab_audio, tab_video = st.tabs([
    "🖼️ Image Forensics",
    "📝 Text Intelligence",
    "🎵 Audio & Voice Clones",
    "🎬 Video Late Fusion",
])

# -----------------------------------------------------------------------------
# TAB 1: IMAGE DETECTOR
# -----------------------------------------------------------------------------
with tab_image:
    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("#### 📤 Upload Visual Media")
        uploaded_image = st.file_uploader(
            "Select an image file",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="img_upload",
        )
        if uploaded_image:
            st.image(uploaded_image, caption=f"Input: {uploaded_image.name}", use_container_width=True)

    with col_res:
        if uploaded_image:
            if st.button("⚡ Run Image Forensic Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing high-frequency residuals and Vision Transformer representations..."):
                    try:
                        validator = FileValidator()
                        raw_data = validator.validate_file(uploaded_image, uploaded_image.name, "image")
                        
                        detector = get_cached_image_detector()
                        detector.threshold_cfg = thresh_cfg
                        result: DetectionResult = detector.classify(raw_data)

                        # Verdict Banner
                        if result.verdict == Verdict.LIKELY_AI:
                            st.markdown(f'<div class="verdict-hero-ai"><div class="verdict-title" style="color:#EF4444;">🔴 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)
                        elif result.verdict == Verdict.LIKELY_HUMAN:
                            st.markdown(f'<div class="verdict-hero-human"><div class="verdict-title" style="color:#10B981;">🟢 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="verdict-hero-uncertain"><div class="verdict-title" style="color:#F59E0B;">🟡 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)

                        st.plotly_chart(render_gauge(result.score, "AI Diffusion Likelihood"), use_container_width=True)

                        # Key Forensic Evidence Tags
                        st.markdown("##### 🔬 Identified Visual Factors")
                        for factor in result.evidence.get("forensic_factors", []):
                            st.markdown(f'<div class="factor-tag">🔹 {factor}</div>', unsafe_allow_html=True)

                        # Technical Metrics Grid
                        freq = result.evidence.get("frequency_analysis", {})
                        c_m1, c_m2, c_m3 = st.columns(3)
                        c_m1.metric("Dimensions", result.evidence.get("dimensions", "N/A"))
                        c_m2.metric("Laplacian Texture Variance", f"{freq.get('laplacian_variance', 0):.1f}")
                        c_m3.metric("HF/LF Spectral Ratio", f"{freq.get('high_low_frequency_ratio', 0):.2f}")

                        st.info(f"⚖️ **Disclaimer:** {result.disclaimer}")

                    except Exception as e:
                        st.error(f"❌ {e}")
        else:
            st.info("Upload an image on the left to start forensic inspection.")

# -----------------------------------------------------------------------------
# TAB 2: TEXT DETECTOR
# -----------------------------------------------------------------------------
with tab_text:
    preset_choice = st.selectbox(
        "Load sample text preset:",
        [
            "-- Select preset or paste custom text below --",
            "ChatGPT Technical Explanation (Database Sharding)",
            "Human Academic Paper Excerpt (Cellular Biology)",
            "Human Reflective Essay (Small Town Community)",
        ],
    )

    preset_map = {
        "ChatGPT Technical Explanation (Database Sharding)": (
            "Database sharding is a horizontal partitioning architecture that separates very large databases "
            "into smaller, faster, and more easily managed parts called shards. Each shard is held on a separate "
            "database server instance, to spread load. Furthermore, by distributing query loads across multiple "
            "physical nodes, applications can achieve massive scalability and eliminate single-point bottlenecks."
        ),
        "Human Academic Paper Excerpt (Cellular Biology)": (
            "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed "
            "to power the cell's biochemical reactions. Chemical energy produced by the mitochondria is stored "
            "in a small molecule called adenosine triphosphate, commonly referred to as ATP, which serves as the "
            "universal energy currency across eukaryotic life."
        ),
        "Human Reflective Essay (Small Town Community)": (
            "Growing up in a small town taught me the importance of community and mutual trust. Whenever there "
            "was a severe storm or unexpected snowfall, neighbors would naturally gather with shovels to clear "
            "each other's driveways without anyone asking. That sense of unspoken solidarity profoundly shaped my character."
        ),
    }

    raw_text = st.text_area(
        "Paste text to inspect (Minimum 50 characters):",
        value=preset_map.get(preset_choice, ""),
        height=160,
        placeholder="Enter essay, article, or generated text...",
    )

    w_count = len(raw_text.strip().split())
    c_count = len(raw_text.strip())
    st.caption(f"**Word Count:** {w_count} | **Character Count:** {c_count} (Min: 50)")

    if st.button("⚡ Run Text Intelligence Analysis", type="primary", use_container_width=True):
        if c_count < 50:
            st.warning("⚠️ Please provide at least 50 characters for reliable sequence classification.")
        else:
            with st.spinner("Analyzing sentence perplexity and transformer representations..."):
                try:
                    detector = get_cached_text_detector()
                    detector.threshold_cfg = thresh_cfg
                    result: DetectionResult = detector.classify(raw_text)

                    if result.verdict == Verdict.LIKELY_AI:
                        st.markdown(f'<div class="verdict-hero-ai"><div class="verdict-title" style="color:#EF4444;">🔴 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)
                    elif result.verdict == Verdict.LIKELY_HUMAN:
                        st.markdown(f'<div class="verdict-hero-human"><div class="verdict-title" style="color:#10B981;">🟢 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="verdict-hero-uncertain"><div class="verdict-title" style="color:#F59E0B;">🟡 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)

                    c_g, c_f = st.columns([1, 1])
                    with c_g:
                        st.plotly_chart(render_gauge(result.score, "AI Generation Likelihood"), use_container_width=True)
                    with c_f:
                        st.markdown("##### 🔬 Syntactic & Readability Factors")
                        for factor in result.evidence.get("readability_factors", []):
                            st.markdown(f'<div class="factor-tag">🔹 {factor}</div>', unsafe_allow_html=True)
                        burst = result.evidence.get("burstiness_metrics", {})
                        st.caption(f"**Burstiness Index:** `{burst.get('burstiness_index', 0)}` | **Length Variance:** `{burst.get('sentence_length_variance', 0)}`")

                    # Sentence-by-Sentence Explainability Highlight Viewer (Google/Microsoft Style)
                    st.markdown("##### 📑 Sentence-by-Sentence Forensic Heatmap")
                    sentences = result.evidence.get("sentence_breakdown", [])
                    for s_item in sentences:
                        p = s_item.get("pattern", "Neutral")
                        css_cls = "sentence-ai" if "AI" in p else ("sentence-human" if "Human" in p else "sentence-neutral")
                        st.markdown(f'<div class="sentence-box {css_cls}"><b>[{p}]</b> {s_item.get("sentence")} <span style="float:right; font-size:0.8rem; color:#94A3B8;">AI: {s_item.get("ai_score", 0)*100:.0f}%</span></div>', unsafe_allow_html=True)

                    st.info(f"⚖️ **Disclaimer:** {result.disclaimer}")

                except Exception as e:
                    st.error(f"❌ {e}")

# -----------------------------------------------------------------------------
# TAB 3: AUDIO DETECTOR
# -----------------------------------------------------------------------------
with tab_audio:
    col_a_up, col_a_res = st.columns([1, 1], gap="large")

    with col_a_up:
        st.markdown("#### 🎙️ Upload Acoustic Media")
        uploaded_audio = st.file_uploader(
            "Select an audio file (WAV, MP3, FLAC, OGG, M4A)",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="aud_upload",
        )
        if uploaded_audio:
            st.audio(uploaded_audio)
            st.caption(f"File: `{uploaded_audio.name}` ({uploaded_audio.size / 1024:.1f} KB)")

    with col_a_res:
        if uploaded_audio:
            if st.button("⚡ Run Audio Deepfake Inspection", type="primary", use_container_width=True):
                with st.spinner("Analyzing spectral harmonics and Wav2Vec2 representations..."):
                    try:
                        validator = FileValidator()
                        raw_data = validator.validate_file(uploaded_audio, uploaded_audio.name, "audio")
                        
                        detector = get_cached_audio_detector()
                        detector.threshold_cfg = thresh_cfg
                        result: DetectionResult = detector.classify(raw_data)

                        if result.verdict == Verdict.LIKELY_AI:
                            st.markdown(f'<div class="verdict-hero-ai"><div class="verdict-title" style="color:#EF4444;">🔴 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)
                        elif result.verdict == Verdict.LIKELY_HUMAN:
                            st.markdown(f'<div class="verdict-hero-human"><div class="verdict-title" style="color:#10B981;">🟢 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="verdict-hero-uncertain"><div class="verdict-title" style="color:#F59E0B;">🟡 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)

                        st.plotly_chart(render_gauge(result.score, "Synthetic Voice / Deepfake Score"), use_container_width=True)

                        c_a1, c_a2, c_a3 = st.columns(3)
                        c_a1.metric("Duration", f"{result.evidence.get('duration_seconds', 0)}s")
                        c_a2.metric("Peak Chunk Risk", f"{result.evidence.get('peak_chunk_score', 0)*100:.1f}%")
                        c_a3.metric("Chunks Evaluated", result.evidence.get("num_chunks_analyzed", 1))

                        st.info(f"⚖️ **Disclaimer:** {result.disclaimer}")

                    except Exception as e:
                        st.error(f"❌ {e}")
        else:
            st.info("Upload an audio file on the left to begin inspection.")

# -----------------------------------------------------------------------------
# TAB 4: VIDEO DETECTOR
# -----------------------------------------------------------------------------
with tab_video:
    col_v_up, col_v_res = st.columns([1, 1], gap="large")

    with col_v_up:
        st.markdown("#### 🎬 Upload Video Stream")
        uploaded_video = st.file_uploader(
            "Select a video file (MP4, MOV, AVI, WebM)",
            type=["mp4", "mov", "avi", "webm"],
            key="vid_upload",
        )
        if uploaded_video:
            st.video(uploaded_video)
            st.caption(f"File: `{uploaded_video.name}` ({uploaded_video.size / (1024*1024):.2f} MB)")

    with col_v_res:
        if uploaded_video:
            if st.button("⚡ Run Multimodal Video Late Fusion", type="primary", use_container_width=True):
                with st.spinner("Extracting uniform keyframes and demuxing audio tracks..."):
                    try:
                        validator = FileValidator()
                        raw_data = validator.validate_file(uploaded_video, uploaded_video.name, "video")
                        
                        detector = get_cached_video_detector()
                        detector.threshold_cfg = thresh_cfg
                        result: DetectionResult = detector.classify(raw_data)

                        if result.verdict == Verdict.LIKELY_AI:
                            st.markdown(f'<div class="verdict-hero-ai"><div class="verdict-title" style="color:#EF4444;">🔴 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)
                        elif result.verdict == Verdict.LIKELY_HUMAN:
                            st.markdown(f'<div class="verdict-hero-human"><div class="verdict-title" style="color:#10B981;">🟢 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="verdict-hero-uncertain"><div class="verdict-title" style="color:#F59E0B;">🟡 {result.verdict.value}</div><div>Confidence: <b>{result.confidence.value}</b> | Latency: <b>{result.processing_time_ms:.0f}ms</b></div></div>', unsafe_allow_html=True)

                        st.plotly_chart(render_gauge(result.score, "Multimodal Fused Risk Score"), use_container_width=True)

                        c_vf, c_vv, c_va = st.columns(3)
                        c_vf.metric("Fused Score", f"{result.score * 100:.1f}%")
                        v_sc = result.evidence.get("visual_score")
                        c_vv.metric("Visual Stream", f"{v_sc * 100:.1f}%" if v_sc is not None else "N/A")
                        a_sc = result.evidence.get("audio_score")
                        c_va.metric("Acoustic Stream", f"{a_sc * 100:.1f}%" if a_sc is not None else "N/A (Silent)")

                        # Frame-by-Frame Scrubbing Table
                        st.markdown("##### 🎞️ Temporal Keyframe Breakdown")
                        frame_data = result.evidence.get("frame_by_frame_analysis", [])
                        if frame_data:
                            df_f = pd.DataFrame(frame_data)
                            st.dataframe(df_f, use_container_width=True, hide_index=True)

                        st.info(f"⚖️ **Disclaimer:** {result.disclaimer}")

                    except Exception as e:
                        st.error(f"❌ {e}")
        else:
            st.info("Upload a video container on the left to begin analysis.")
