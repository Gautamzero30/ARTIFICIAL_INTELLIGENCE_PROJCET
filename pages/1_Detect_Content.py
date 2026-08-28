"""
Authentica AI — Interactive Multimodal Detection Interface.
"""
from pathlib import Path
import streamlit as st

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
    page_title="Detect Content — Authentica AI",
    page_icon="🔍",
    layout="wide",
)

# Cached Detector Singletons
@st.cache_resource(show_spinner="Initializing Vision Transformer image model...")
def get_image_detector() -> ImageDetector:
    detector = ImageDetector()
    detector.load_model()
    return detector


@st.cache_resource(show_spinner="Initializing RoBERTa text model...")
def get_text_detector() -> TextDetector:
    detector = TextDetector()
    detector.load_model()
    return detector


st.title("🔍 Multimodal Content Detection")
st.write("Submit content to estimate the probability of AI generation versus authentic human creation.")

# Tabbed Interface
tab_image, tab_text, tab_audio, tab_video = st.tabs([
    "🖼️ Image Detection",
    "📝 Text Detection",
    "🎵 Audio Detection (Phase 7)",
    "🎬 Video Late Fusion (Phase 8)",
])

# -----------------------------------------------------------------------------
# TAB 1: IMAGE DETECTION
# -----------------------------------------------------------------------------
with tab_image:
    st.subheader("🖼️ Image AI Detection")
    st.write("Analyzes high-frequency diffusion artifacts, textures, and structural patterns.")

    col_up, col_res = st.columns([1, 1])

    with col_up:
        uploaded_image = st.file_uploader(
            "Upload an image (JPG, PNG, WebP, BMP)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="image_uploader",
        )

        if uploaded_image is not None:
            st.image(uploaded_image, caption=f"Uploaded: {uploaded_image.name}", use_container_width=True)

    with col_res:
        if uploaded_image is not None:
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Processing image through Vision Transformer..."):
                    try:
                        # 1. Security & MIME validation
                        validator = FileValidator()
                        validated_bytes = validator.validate_file(
                            file_obj=uploaded_image,
                            filename=uploaded_image.name,
                            modality="image",
                        )

                        # 2. Run Image Detector
                        detector = get_image_detector()
                        result: DetectionResult = detector.classify(validated_bytes)

                        # 3. Render Results
                        st.markdown("### 📋 Analysis Result")

                        # Color-coded verdict badge
                        if result.verdict == Verdict.LIKELY_AI:
                            st.error(f"### 🔴 Verdict: {result.verdict.value}")
                        elif result.verdict == Verdict.LIKELY_HUMAN:
                            st.success(f"### 🟢 Verdict: {result.verdict.value}")
                        else:
                            st.warning(f"### 🟡 Verdict: {result.verdict.value}")

                        c_sc1, c_sc2, c_sc3 = st.columns(3)
                        c_sc1.metric("AI Detection Score", f"{result.score * 100:.1f}%")
                        c_sc2.metric("Confidence Level", result.confidence.value)
                        c_sc3.metric("Latency", f"{result.processing_time_ms:.0f} ms")

                        st.progress(float(result.score))

                        # Technical Evidence Expander
                        with st.expander("🔍 Detailed Technical Evidence", expanded=False):
                            st.json(result.evidence)
                            st.caption(f"Model ID: `{result.model_name}` | Device: `{result.evidence.get('device', 'cpu')}`")

                        # Standard Disclaimer
                        st.info(f"⚖️ **Disclaimer:** {result.disclaimer}")

                    except (ValidationError, FileSecurityError, CorruptedFileError, UnsupportedFormatError) as e:
                        st.error(f"❌ **Validation Error:** {e}")
                    except Exception as e:
                        st.error(f"❌ **Inference Error:** An unexpected error occurred during processing: {e}")
        else:
            st.info("👆 Please upload an image file on the left to begin analysis.")

# -----------------------------------------------------------------------------
# TAB 2: TEXT DETECTION
# -----------------------------------------------------------------------------
with tab_text:
    st.subheader("📝 Text AI Detection")
    st.write("Analyzes vocabulary patterns, syntax, and sentence flow using RoBERTa sequence classification.")

    # Preset sample selector for easy demonstration
    sample_choice = st.selectbox(
        "Load sample text preset (or paste your own below):",
        [
            "-- Select a preset or type below --",
            "Human Academic Sample (Photosynthesis)",
            "ChatGPT Response Sample (Database Optimization)",
            "Short Ambiguous Text (<50 characters test)",
        ],
    )

    preset_text = ""
    if sample_choice == "Human Academic Sample (Photosynthesis)":
        preset_text = (
            "Photosynthesis is a biological process utilized by plants and other photosynthetic organisms "
            "to convert light energy into chemical energy that can later be released to fuel the organism's "
            "metabolic activities. This chemical energy is stored in carbohydrate molecules such as sugars, "
            "which are synthesized from carbon dioxide and water."
        )
    elif sample_choice == "ChatGPT Response Sample (Database Optimization)":
        preset_text = (
            "Certainly! To optimize a database query, it is essential to first analyze the execution plan "
            "and ensure that appropriate indexes are created on frequently queried columns. Furthermore, "
            "minimizing the use of subqueries and replacing them with efficient inner or outer joins can "
            "significantly reduce execution latency and improve database throughput."
        )
    elif sample_choice == "Short Ambiguous Text (<50 characters test)":
        preset_text = "This text is too short."

    text_input = st.text_area(
        "Enter or paste text to analyze (minimum 50 characters):",
        value=preset_text,
        height=180,
        placeholder="Paste an article, essay, or paragraph here...",
    )

    char_count = len(text_input.strip())
    word_count = len(text_input.strip().split())
    st.caption(f"Character Count: **{char_count}** | Word Count: **{word_count}** (Minimum required: 50 characters)")

    if st.button("🚀 Analyze Text", type="primary", use_container_width=True):
        if not text_input or char_count < 50:
            st.warning("⚠️ Text must contain at least 50 characters for reliable analysis.")
        else:
            with st.spinner("Processing text through RoBERTa transformer..."):
                try:
                    detector = get_text_detector()
                    result: DetectionResult = detector.classify(text_input)

                    st.markdown("### 📋 Analysis Result")

                    if result.verdict == Verdict.LIKELY_AI:
                        st.error(f"### 🔴 Verdict: {result.verdict.value}")
                    elif result.verdict == Verdict.LIKELY_HUMAN:
                        st.success(f"### 🟢 Verdict: {result.verdict.value}")
                    else:
                        st.warning(f"### 🟡 Verdict: {result.verdict.value}")

                    c_sc1, c_sc2, c_sc3 = st.columns(3)
                    c_sc1.metric("AI Detection Score", f"{result.score * 100:.1f}%")
                    c_sc2.metric("Confidence Level", result.confidence.value)
                    c_sc3.metric("Latency", f"{result.processing_time_ms:.0f} ms")

                    st.progress(float(result.score))

                    if "language_warning" in result.evidence:
                        st.warning(f"🌐 **Language Note:** {result.evidence['language_warning']}")

                    with st.expander("🔍 Detailed Technical Evidence & Chunking", expanded=False):
                        st.json(result.evidence)
                        st.caption(f"Model ID: `{result.model_name}` | Analyzed in {result.evidence.get('num_chunks_analyzed', 1)} chunk(s)")

                    st.info(f"⚖️ **Disclaimer:** {result.disclaimer}")

                except InsufficientContentError as e:
                    st.warning(f"⚠️ {e}")
                except Exception as e:
                    st.error(f"❌ **Inference Error:** {e}")

# Cached Audio Detector Singleton
@st.cache_resource(show_spinner="Initializing Wav2Vec2 audio deepfake model...")
def get_audio_detector() -> AudioDetector:
    detector = AudioDetector()
    detector.load_model()
    return detector


# -----------------------------------------------------------------------------
# TAB 3: AUDIO DETECTION
# -----------------------------------------------------------------------------
with tab_audio:
    st.subheader("🎵 Synthetic Audio & Voice Clone Detection")
    st.write("Analyzes acoustic artifacts, vocoder harmonics, and phase coherence using Wav2Vec2.")

    col_aud_up, col_aud_res = st.columns([1, 1])

    with col_aud_up:
        uploaded_audio = st.file_uploader(
            "Upload an audio file (WAV, MP3, FLAC, OGG, M4A)",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="audio_uploader",
        )

        if uploaded_audio is not None:
            st.audio(uploaded_audio)
            st.caption(f"File: `{uploaded_audio.name}` ({uploaded_audio.size / 1024:.1f} KB)")

    with col_aud_res:
        if uploaded_audio is not None:
            if st.button("🚀 Analyze Audio", type="primary", use_container_width=True):
                with st.spinner("Processing audio through Wav2Vec2 classifier..."):
                    try:
                        validator = FileValidator()
                        validated_bytes = validator.validate_file(
                            file_obj=uploaded_audio,
                            filename=uploaded_audio.name,
                            modality="audio",
                        )

                        detector = get_audio_detector()
                        result: DetectionResult = detector.classify(validated_bytes)

                        st.markdown("### 📋 Analysis Result")

                        if result.verdict == Verdict.LIKELY_AI:
                            st.error(f"### 🔴 Verdict: {result.verdict.value}")
                        elif result.verdict == Verdict.LIKELY_HUMAN:
                            st.success(f"### 🟢 Verdict: {result.verdict.value}")
                        else:
                            st.warning(f"### 🟡 Verdict: {result.verdict.value}")

                        c_sc1, c_sc2, c_sc3 = st.columns(3)
                        c_sc1.metric("AI Detection Score", f"{result.score * 100:.1f}%")
                        c_sc2.metric("Confidence Level", result.confidence.value)
                        c_sc3.metric("Latency", f"{result.processing_time_ms:.0f} ms")

                        st.progress(float(result.score))

                        with st.expander("🔍 Acoustic Evidence & Chunk Analysis", expanded=False):
                            st.json(result.evidence)
                            st.caption(f"Model ID: `{result.model_name}` | Duration: {result.evidence.get('duration_seconds', 0)}s | Chunks: {result.evidence.get('num_chunks_analyzed', 1)}")

                        st.info(f"⚖️ **Disclaimer:** {result.disclaimer}")

                    except (ValidationError, FileSecurityError, CorruptedFileError, UnsupportedFormatError) as e:
                        st.error(f"❌ **Validation Error:** {e}")
                    except Exception as e:
                        st.error(f"❌ **Inference Error:** {e}")
        else:
            st.info("👆 Please upload an audio file on the left to begin analysis.")

# Cached Video Detector Singleton
@st.cache_resource(show_spinner="Initializing Multimodal Video Detector...")
def get_video_detector() -> VideoDetector:
    from src.detectors.video import VideoDetector
    detector = VideoDetector()
    detector.load_model()
    return detector


# -----------------------------------------------------------------------------
# TAB 4: VIDEO DETECTION
# -----------------------------------------------------------------------------
with tab_video:
    st.subheader("🎬 Video Multimodal Late Fusion Detection")
    st.write("Extracts temporal keyframes (ViT) and acoustic audio tracks (Wav2Vec2), combining both via weighted late fusion.")

    col_vid_up, col_vid_res = st.columns([1, 1])

    with col_vid_up:
        uploaded_video = st.file_uploader(
            "Upload a video file (MP4, MOV, AVI, WebM)",
            type=["mp4", "mov", "avi", "webm"],
            key="video_uploader",
        )

        if uploaded_video is not None:
            st.video(uploaded_video)
            st.caption(f"File: `{uploaded_video.name}` ({uploaded_video.size / (1024*1024):.2f} MB)")

    with col_vid_res:
        if uploaded_video is not None:
            if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
                with st.spinner("Extracting keyframes and demuxing audio track..."):
                    try:
                        validator = FileValidator()
                        validated_bytes = validator.validate_file(
                            file_obj=uploaded_video,
                            filename=uploaded_video.name,
                            modality="video",
                        )

                        detector = get_video_detector()
                        result: DetectionResult = detector.classify(validated_bytes)

                        st.markdown("### 📋 Multimodal Analysis Result")

                        if result.verdict == Verdict.LIKELY_AI:
                            st.error(f"### 🔴 Verdict: {result.verdict.value}")
                        elif result.verdict == Verdict.LIKELY_HUMAN:
                            st.success(f"### 🟢 Verdict: {result.verdict.value}")
                        else:
                            st.warning(f"### 🟡 Verdict: {result.verdict.value}")

                        c_fused, c_vis, c_aud = st.columns(3)
                        c_fused.metric("Fused AI Score", f"{result.score * 100:.1f}%")
                        
                        v_sc = result.evidence.get("visual_score")
                        c_vis.metric("Visual Stream Score", f"{v_sc * 100:.1f}%" if v_sc is not None else "N/A")
                        
                        a_sc = result.evidence.get("audio_score")
                        c_aud.metric("Audio Stream Score", f"{a_sc * 100:.1f}%" if a_sc is not None else "N/A (Silent)")

                        st.progress(float(result.score))

                        # Fusion Mode & Peak Anomaly
                        c_m1, c_m2 = st.columns(2)
                        c_m1.caption(f"**Fusion Mode:** `{result.evidence.get('fusion_mode', 'dual_stream')}`")
                        peak_anom = result.evidence.get("peak_frame_anomaly", {})
                        c_m2.caption(f"**Peak Frame Anomaly:** {peak_anom.get('peak_score', 0)*100:.1f}% at `{peak_anom.get('timestamp_sec', 0)}s`")

                        # Frame-by-Frame Table Expander
                        with st.expander("🎞️ Frame-by-Frame Visual Breakdown", expanded=False):
                            frame_data = result.evidence.get("frame_by_frame_analysis", [])
                            if frame_data:
                                import pandas as pd
                                df_frames = pd.DataFrame(frame_data)
                                st.dataframe(df_frames, use_container_width=True)

                        with st.expander("🔍 Complete Technical Evidence Payload", expanded=False):
                            st.json(result.evidence)

                        st.info(f"⚖️ **Disclaimer:** {result.disclaimer}")

                    except (ValidationError, FileSecurityError, CorruptedFileError, UnsupportedFormatError) as e:
                        st.error(f"❌ **Validation Error:** {e}")
                    except Exception as e:
                        st.error(f"❌ **Inference Error:** {e}")
        else:
            st.info("👆 Please upload a video file on the left to begin analysis.")
