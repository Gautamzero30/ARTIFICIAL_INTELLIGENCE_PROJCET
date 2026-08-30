"""
Authentica AI — AI-Generated Visual Content Detection Studio.
Focused exclusively on Image and Video Forensics.
"""
from pathlib import Path
from PIL import Image
import streamlit as st

from src.core.config import load_settings
from src.detectors.base import DetectionResult
from src.detectors.image import ImageDetector
from src.detectors.video import VideoDetector
from src.ui.components.detection_score import render_detection_score
from src.ui.components.disclaimer import render_disclaimer
from src.ui.components.evidence_panel import render_image_evidence, render_video_evidence
from src.ui.components.media_metadata import render_image_metadata, render_video_metadata
from src.ui.components.result_card import render_result_card
from src.utils.file_validator import FileValidator

st.set_page_config(
    page_title="Our Functionalities — Authentica AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .brand-header {
        border-bottom: 2px solid #334155;
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem;
    }
    .brand-title {
        font-family: monospace;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #F8FAFC;
        text-transform: uppercase;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.9rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    .section-title {
        font-family: monospace;
        font-size: 0.92rem;
        font-weight: 800;
        color: #F8FAFC;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 3.5px solid #38BDF8;
        padding-left: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_image_detector() -> ImageDetector:
    _settings = load_settings()
    det = ImageDetector(threshold_config=_settings.thresholds)
    det.load_model()
    return det


@st.cache_resource(show_spinner=False)
def get_cached_video_detector() -> VideoDetector:
    _settings = load_settings()
    det = VideoDetector(threshold_config=_settings.video_thresholds)
    det.load_model()
    return det


# Top Header
st.markdown(
    """<div class="brand-header">
<div class="brand-title">AUTHENTICA AI</div>
<div class="brand-subtitle">AI-Generated Visual Content Detection Studio</div>
</div>""",
    unsafe_allow_html=True,
)

# Sidebar Threshold Documentation
with st.sidebar:
    st.markdown("### ⚙️ Threshold Governance")
    st.caption("**Three-Way Decision Policy:**\n• `< 40%`: LIKELY HUMAN-CREATED\n• `40% – 45%`: UNCERTAIN\n• `&ge; 45%`: LIKELY AI-GENERATED")
    st.markdown("---")
    st.markdown("### 🔒 Security Verification")
    st.caption("✅ Deep Magic Byte Signature Check\n✅ MIME & Container Integrity\n✅ Sandboxed Processing")

# Main Modality Tabs: Image and Video ONLY
tab_image, tab_video = st.tabs([
    "🖼️ Image Analysis",
    "🎬 Video Analysis",
])


# =============================================================================
# TAB 1: IMAGE ANALYSIS
# =============================================================================
with tab_image:
    st.markdown('<div class="section-title">IMAGE FORENSIC ANALYSIS</div>', unsafe_allow_html=True)
    col_img_up, col_img_res = st.columns([1, 1], gap="large")

    with col_img_up:
        uploaded_image = st.file_uploader(
            "Select an image file (JPG, JPEG, PNG, WEBP, BMP)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="img_upload",
            help="Maximum file size: 25 MB",
        )
        st.caption("📏 **Maximum file size: 25 MB** • **Supported formats:** JPG, JPEG, PNG, WEBP, BMP")

        if uploaded_image:
            try:
                pil_img_input = Image.open(uploaded_image)
                st.image(pil_img_input, caption=f"Input Image: {uploaded_image.name}", use_container_width=True)
                
                w, h = pil_img_input.size
                render_image_metadata(
                    filename=uploaded_image.name,
                    size_bytes=uploaded_image.size,
                    dimensions=f"{w} x {h} px",
                    format_name=pil_img_input.format or Path(uploaded_image.name).suffix.lstrip("."),
                )

                if st.session_state.get("img_filename") != uploaded_image.name:
                    st.session_state.pop("img_result", None)

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("⚡ Run Image Analysis", type="primary", use_container_width=True):
                    with st.spinner("Analyzing Vision Transformer representations..."):
                        try:
                            validator = FileValidator()
                            raw_data = validator.validate_file(uploaded_image, uploaded_image.name, "image")
                            
                            detector = get_cached_image_detector()
                            result: DetectionResult = detector.classify(raw_data)
                            
                            st.session_state["img_result"] = result
                            st.session_state["img_pil"] = pil_img_input
                            st.session_state["img_filename"] = uploaded_image.name
                        except Exception as e:
                            st.error(f"❌ {e}")

            except Exception as e:
                st.error(f"❌ Could not load image preview: {e}")

    with col_img_res:
        if uploaded_image and "img_result" in st.session_state:
            res: DetectionResult = st.session_state["img_result"]
            pil_in: Image.Image = st.session_state.get("img_pil", pil_img_input)

            # Result Card
            render_result_card(res)

            # Decision Scale
            render_detection_score(
                score=res.score,
                lower_threshold=0.40,
                upper_threshold=0.45,
                verdict=res.verdict,
            )

            # Evidence & Technical Details
            render_image_evidence(res, pil_in)

            # Disclaimer
            render_disclaimer()

        elif not uploaded_image:
            st.info("Upload an image on the left to begin forensic verification.")


# =============================================================================
# TAB 2: VIDEO ANALYSIS
# =============================================================================
with tab_video:
    st.markdown('<div class="section-title">MULTIMODAL VIDEO ANALYSIS</div>', unsafe_allow_html=True)
    col_vid_up, col_vid_res = st.columns([1, 1], gap="large")

    with col_vid_up:
        uploaded_video = st.file_uploader(
            "Select a video file (MP4, MOV, AVI, WebM)",
            type=["mp4", "mov", "avi", "webm"],
            key="vid_upload",
            help="Maximum file size: 250 MB • Maximum duration: 50 seconds",
        )
        st.caption("📏 **Maximum file size: 250 MB • Maximum duration: 50 seconds** • **Supported formats:** MP4, MOV, AVI, WebM")

        if uploaded_video:
            st.video(uploaded_video)

            if st.session_state.get("vid_filename") != uploaded_video.name:
                st.session_state.pop("vid_result", None)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Run Video Analysis", type="primary", use_container_width=True):
                with st.spinner("Extracting uniform keyframes and evaluating multimodal signals..."):
                    try:
                        validator = FileValidator()
                        raw_data = validator.validate_file(uploaded_video, uploaded_video.name, "video")
                        
                        detector = get_cached_video_detector()
                        result: DetectionResult = detector.classify(raw_data)

                        st.session_state["vid_result"] = result
                        st.session_state["vid_filename"] = uploaded_video.name
                    except Exception as e:
                        st.error(f"❌ {e}")

    with col_vid_res:
        if uploaded_video and "vid_result" in st.session_state:
            res: DetectionResult = st.session_state["vid_result"]
            v_meta = res.evidence.get("video_metadata", {})

            # Container Metadata
            render_video_metadata(
                filename=uploaded_video.name,
                size_bytes=uploaded_video.size,
                duration_sec=v_meta.get("duration_seconds", 0.0),
                resolution=v_meta.get("resolution", "N/A"),
                fps=v_meta.get("fps", 0.0),
                total_frames=v_meta.get("total_frames", 0),
            )

            # Result Card
            render_result_card(res)

            # Decision Scale
            render_detection_score(
                score=res.score,
                lower_threshold=0.40,
                upper_threshold=0.45,
                verdict=res.verdict,
            )

            # Evidence & Keyframe Profile
            render_video_evidence(res)

            # Disclaimer
            render_disclaimer()

        elif not uploaded_video:
            st.info("Upload a video container on the left to begin multimodal analysis.")
