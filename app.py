"""
Authentica AI — Enterprise AI Content Safety & Multimodal Detection Portal.
Inspired by Microsoft Azure AI Content Safety and Google Cloud Vertex AI Design Language.
"""
import streamlit as st

st.set_page_config(
    page_title="Authentica AI — Multimodal Content Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS (Google / Microsoft Azure Design Language)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-tagline {
        font-size: 1.15rem;
        color: #94A3B8;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.6;
    }
    .feature-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease-in-out;
        height: 100%;
    }
    .feature-card:hover {
        border-color: #60A5FA;
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.4);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.8rem;
    }
    .feature-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.4rem;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .disclaimer-banner {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 1.2rem;
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
        <div class="status-badge">● Production Capstone Architecture</div>
        <div class="hero-title">Authentica AI</div>
        <div class="hero-tagline">
            Next-generation Multimodal AI Content Intelligence platform. Analyzes Images, Text, Audio, 
            and Video using Vision Transformers, RoBERTa sequence models, Wav2Vec2 acoustic architectures, 
            and late fusion decision engines.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Modality Capabilities Grid
st.markdown("### ⚡ Multimodal Forensic Engines")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🖼️</div>
            <div class="feature-title">Image Forensics</div>
            <div class="feature-desc">
                Vision Transformer (ViT) architecture combined with 2D Fourier high-frequency residual analysis and Laplacian texture metrics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">Text Intelligence</div>
            <div class="feature-desc">
                RoBERTa transformer with sentence-level Perplexity/Burstiness heatmaps and multi-chunk sliding window analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🎵</div>
            <div class="feature-title">Audio & Voice Clones</div>
            <div class="feature-desc">
                Wav2Vec2 sequence classifier trained on modern neural voice clones (ElevenLabs, Polly) with Mel-spectrogram forensic analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🎬</div>
            <div class="feature-title">Video Late Fusion</div>
            <div class="feature-desc">
                Temporal uniform keyframe sampling combined with demuxed audio analysis and weighted late fusion with silent-video fallback.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Governance & Quickstart
col_nav1, col_nav2 = st.columns([3, 2])

with col_nav1:
    st.markdown("### 🎯 Decision Framework & Policy")
    st.write(
        """
        Authentica AI operates strictly within an **auditable probabilistic governance model**:
        * **LIKELY AI-GENERATED ($S_{\\text{AI}} \\ge 0.65$):** Multi-factor synthetic markers detected with high statistical confidence.
        * **LIKELY HUMAN-CREATED ($S_{\\text{AI}} \\le 0.35$):** Natural organic features, idiosyncratic styles, or optical lens signatures.
        * **UNCERTAIN ($0.35 < S_{\\text{AI}} < 0.65$):** Ambiguous, heavily compressed, or hybrid human-edited artifacts.
        """
    )

with col_nav2:
    st.markdown("### 🧭 Portal Navigation")
    st.markdown(
        """
        * **[🔍 Launch Detection Studio](1_Detect_Content):** Test Images, Text, Audio, or Video live.
        * **[📊 View Performance Dashboard](2_Model_Performance):** Review real ROC curves, confusion matrices, and metrics.
        * **[ℹ️ System Architecture & Docs](3_About):** Read technical model cards and governance guidelines.
        """
    )

st.markdown(
    """
    <div class="disclaimer-banner">
        <strong>⚖️ Standard Probabilistic Notice:</strong> 
        Outputs generated by Authentica AI represent AI-based probabilistic estimates and do not constitute absolute forensic proof. 
        Detection reliability depends on content quality, compression, and generator novelty.
    </div>
    """,
    unsafe_allow_html=True,
)
