"""
Authentica AI — Enterprise AI Visual Media Forensics Portal.
Professional multimodal detection platform for Images and Video.
"""
import streamlit as st

st.set_page_config(
    page_title="Authentica AI — Visual Content Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.8rem 2.4rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
    }
    .hero-badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #F1F5F9;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .hero-tagline {
        font-size: 1.02rem;
        color: #94A3B8;
        font-weight: 400;
        max-width: 820px;
        line-height: 1.65;
    }
    .media-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.6rem;
        height: 100%;
        transition: all 0.2s ease-in-out;
    }
    .media-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.4);
    }
    .media-icon {
        font-size: 2.2rem;
        margin-bottom: 0.6rem;
    }
    .media-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.4rem;
    }
    .media-desc {
        font-size: 0.9rem;
        color: #94A3B8;
        line-height: 1.55;
    }
    .media-limits {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.1);
        padding: 0.35rem 0.6rem;
        border-radius: 6px;
        margin-top: 0.8rem;
        display: inline-block;
    }
    .step-box {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        height: 100%;
    }
    .step-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 800;
        color: #38BDF8;
        margin-bottom: 0.3rem;
    }
    .step-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.25rem;
    }
    .step-desc {
        font-size: 0.84rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    .limitation-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-left: 4px solid #F59E0B;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 2rem;
        color: #CBD5E1;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">Authentica AI Forensics</div>
        <div class="hero-title">Authentica AI</div>
        <div class="hero-subtitle">AI-Generated Visual Content Detection Studio</div>
        <div class="hero-tagline">
            Analyze visual media for signals associated with AI-generated and synthetic content using modern 
            Vision Transformers, spatial frequency residual analysis, and multimodal late fusion.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation CTA Buttons
col_cta1, col_cta2, _ = st.columns([1.8, 1.8, 3.4])
with col_cta1:
    st.page_link("pages/1_Our_Functionalities.py", label="🔍 Our Functionalities", icon="⚡", use_container_width=True)
with col_cta2:
    st.page_link("pages/2_About.py", label="ℹ️ Learn How It Works", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Supported Media Section
st.markdown("### ⚡ Supported Visual Media")
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown(
        """
        <div class="media-card">
            <div class="media-icon">🖼️</div>
            <div class="media-title">AI Image Forensics</div>
            <div class="media-desc">
                High-resolution Vision Transformer (ViT) representation analysis combined with 2D Fourier high-frequency residual metrics and Laplacian surface variance.
            </div>
            <div class="media-limits">Max file size: 25 MB • JPG, PNG, WEBP, BMP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="media-card">
            <div class="media-icon">🎬</div>
            <div class="media-title">Multimodal Video Analysis</div>
            <div class="media-desc">
                Uniform temporal keyframe extraction combined with demuxed acoustic stream analysis and late fusion decision classification with silent-video fallback.
            </div>
            <div class="media-limits">Max file size: 250 MB • Max duration: 50 seconds • MP4, MOV, AVI, WEBM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# How It Works Workflow
st.markdown("### 🔄 How It Works")
s1, s2, s3, s4 = st.columns(4, gap="medium")

with s1:
    st.markdown(
        """
        <div class="step-box">
            <div class="step-num">01</div>
            <div class="step-title">Upload Media</div>
            <div class="step-desc">Select an image or video file meeting the supported format and size restrictions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        """
        <div class="step-box">
            <div class="step-num">02</div>
            <div class="step-title">Model Inference</div>
            <div class="step-desc">Deep neural backbones evaluate spatial residuals, attention maps, and keyframes.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s3:
    st.markdown(
        """
        <div class="step-box">
            <div class="step-num">03</div>
            <div class="step-title">Review Score</div>
            <div class="step-desc">Inspect the continuous AI Detection Score mapped against the 40%–45% decision boundaries.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s4:
    st.markdown(
        """
        <div class="step-box">
            <div class="step-num">04</div>
            <div class="step-title">Interpret Result</div>
            <div class="step-desc">Review the structured forensic verdict, detection strength, and technical details.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Important Limitations Notice
st.markdown(
    """
    <div class="limitation-box">
        <strong style="color: #FDE68A;">⚠️ Important Operational Limitation:</strong><br>
        AI-content detection is probabilistic. Results can be affected by heavy compression, digital post-processing, 
        novel generators unseen during training, and model architecture boundaries. Outputs represent statistical likelihood 
        estimates and should not be used as the sole determinant for punitive or legal actions.
    </div>
    """,
    unsafe_allow_html=True,
)
