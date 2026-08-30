"""
Authentica AI — System Architecture, Candidate Models, and Responsible AI Governance.
"""
import streamlit as st

st.set_page_config(
    page_title="Architecture & Governance — Authentica AI",
    page_icon="ℹ️",
    layout="wide",
)

st.title("ℹ️ Architecture & Responsible AI Governance")
st.markdown("Technical specifications, model backbones, decision policy, and ethical limitations.")

# System Architecture Flow
st.markdown("### 🏗️ Visual Forensics Pipeline Architecture")
st.markdown(
    """
    ```text
                           AUTHENTICA AI FORENSICS ENGINE
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
                 🖼️ IMAGE                                  🎬 VIDEO
          Vision Transformer (ViT)                    Uniform Keyframe Demuxer
           (umm-maybe/AI-image)                                │
                     │                                ┌────────┴────────┐
             RGB & Spatial Norm                       ▼                 ▼
                     │                            Keyframes           Acoustic
          High-Frequency 2D FFT                     (ViT)            (Wav2Vec2)
          Residual & Laplacian                        │                 │
                     │                                └────────┬────────┘
                     │                                         ▼
                     │                               Multimodal Late Fusion
                     │                               (Dual-Stream / Fallback)
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                         Deterministic Decision Engine
                         • S_AI < 0.40  ──► LIKELY HUMAN-CREATED
                         • 0.40 <= S_AI <= 0.45 ──► UNCERTAIN
                         • S_AI > 0.45  ──► LIKELY AI-GENERATED
                                          │
                                          ▼
                         Structured Verdict & Disclaimers
    ```
    """
)

# Decision Threshold Methodology
st.markdown("---")
st.markdown("### ⚖️ Decision Threshold Policy")
st.markdown(
    r"""
    Authentica AI enforces an explicit, deterministic 3-way decision policy:

    * **LIKELY HUMAN-CREATED ($S_{\text{AI}} < 0.40$):** 
      Feature distributions align with optical lens characteristics, camera sensor grain, and natural spatial frequency profiles.
    * **UNCERTAIN ($0.40 \le S_{\text{AI}} \le 0.45$):** 
      Boundary zone where statistical representations do not demonstrate decisive separation. Avoids unproven assumptions of content composition.
    * **LIKELY AI-GENERATED ($S_{\text{AI}} \ge 0.45$):** 
      Neural synthesis artifacts, diffusion high-frequency anomalies, or keyframe anomalies detected with significant statistical confidence.
    """
)


# Model Registry
st.markdown("---")
st.markdown("### 🤖 Candidate Model Registry")

with st.expander("🔹 Vision Transformer AI-Image Detector (`umm-maybe/AI-image-detector`)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Modality:** `Image`")
        st.write("**Architecture:** `Vision Transformer (ViT-Base/16-224)`")
        st.write("**Task Domain:** `Binary Image Classification (Artificial vs Human)`")
        st.write("**Model Size:** `~343 MB`")
        st.write("**License:** `MIT`")
    with c2:
        st.write("**Training Domain:** `Midjourney, Stable Diffusion v1.4/v1.5, DALL-E 2, Human Photos`")
        st.write("**Input Specification:** `RGB 224x224 Normalized Tensor`")
        st.write("**Output Format:** `Softmax AI Detection Score`")

with st.expander("🔹 Multimodal Video Late Fusion Engine (`Authentica-Multimodal-Late-Fusion`)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Modality:** `Video`")
        st.write("**Architecture:** `Temporal Keyframe ViT + Acoustic Stream Late Fusion`")
        st.write("**Sampling Strategy:** `Uniform Temporal Sampling (6-8 Keyframes)`")
        st.write("**Default Fusion Weights:** `Visual: 0.60 | Acoustic: 0.40`")
        st.write("**Silent Video Fallback:** `100% Visual Weight (Auto-detected)`")
    with c2:
        st.write("**Supported Formats:** `MP4, MOV, AVI, WebM`")
        st.write("**Max Duration:** `50 Seconds`")
        st.write("**Max File Size:** `250 MB`")
        st.write("**Output Format:** `Fused Detection Score`")

# Responsible AI & Limitations
st.markdown("---")
st.markdown("### 📜 Responsible AI Principles & Known Limitations")
st.markdown(
    """
    1. **Probabilistic Nature:** Authentica AI computes statistical likelihood estimates based on observed high-frequency residuals and learned representations. It does not provide absolute cryptographic proof.
    2. **Compression & Social Media Re-encoding:** Heavy JPEG/H.264 compression, downsampling, and aggressive noise reduction can dampen high-frequency residuals and shift detection scores.
    3. **Generator Generalization:** Detectors demonstrate highest fidelity on architectures represented in their training data. Novel, unseen foundation models may exhibit domain shift.
    4. **Academic & Ethical Purpose:** Engineered for academic research, content provenance screening, and trust & safety exploration. It should not be used as the sole determinant for punitive decisions.
    """
)
