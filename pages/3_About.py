"""
Authentica AI — About, Methodology, and Ethical Limitations.
"""
import streamlit as st
from src.utils.model_registry import list_registered_models

st.set_page_config(
    page_title="About — Authentica AI",
    page_icon="ℹ️",
    layout="wide",
)

st.title("ℹ️ About Authentica AI")
st.markdown("### Multimodal AI-Generated Content Detection System")

st.write(
    """
    **Authentica AI** is an academic capstone project engineered to address the rapid rise of 
    generative artificial intelligence across visual, textual, acoustic, and audiovisual mediums. 
    Rather than claiming infallible forensic certainty, the system emphasizes **probabilistic estimation**, 
    **transparent confidence intervals**, and **rigorous scientific evaluation**.
    """
)

# -----------------------------------------------------------------------------
# SYSTEM ARCHITECTURE
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🏗️ System Architecture & Multimodal Pipeline")

st.markdown(
    """
    ```text
                        AUTHENTICA AI MULTIMODAL PIPELINE
                                       │
         ┌─────────────────┬───────────┴───────────┬─────────────────┐
         ▼                 ▼                       ▼                 ▼
     🖼️ IMAGE           📝 TEXT                 🎵 AUDIO          🎬 VIDEO
    Vision Trans.      RoBERTa Seq.           Wav2Vec2 Seq.     Decoupled Stream
   (ViT-Base/16)       (HC3 Domain)           (Voice Clones)     Late Fusion
         │                 │                       │                 │
         │                 │                       │           ┌─────┴─────┐
         │                 │                       │           ▼           ▼
         │                 │                       │        Frames       Audio
         │                 │                       │         (ViT)     (Wav2Vec2)
         │                 │                       │           │           │
         │                 │                       │           └─────┬─────┘
         │                 │                       │                 ▼
         │                 │                       │          Weighted Fusion
         │                 │                       │                 │
         └─────────────────┼───────────────────────┼─────────────────┘
                           │
                           ▼
               Decision Classification Engine
               - S_AI >= 0.65  -> LIKELY AI-GENERATED
               - S_AI <= 0.35  -> LIKELY HUMAN-CREATED
               - 0.35 < S < 0.65 -> UNCERTAIN
                           │
                           ▼
              Structured Result & Disclaimers
    ```
    """
)

# -----------------------------------------------------------------------------
# CANDIDATE MODEL REGISTRY
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🤖 Candidate Model Registry & Specifications")

models = list_registered_models()
for m in models:
    with st.expander(f"🔹 {m.display_name} (`{m.model_id}`)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Modality:** {m.modality.capitalize()}")
            st.write(f"**Architecture:** {m.architecture}")
            st.write(f"**Target Task:** {m.task}")
            st.write(f"**License:** {m.license}")
            st.write(f"**Model Size:** ~{m.size_mb} MB")
        with c2:
            st.write(f"**Training Domain:** {m.training_domain}")
            st.write(f"**Input Contract:** {m.input_format}")
            st.write(f"**Output Format:** {m.output_format}")
            if m.notes:
                st.info(f"📌 **Notes:** {m.notes}")

# -----------------------------------------------------------------------------
# ETHICAL & TECHNICAL LIMITATIONS
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("⚖️ Ethical Considerations & Technical Limitations")

st.markdown(
    """
    1. **Probabilistic Nature:** AI detection algorithms analyze statistical artifacts and representation distributions. They do not possess ontological awareness and can produce both **false positives** (flagging human creative writing/photos as AI) and **false negatives** (missing sophisticated or edited AI content).
    2. **Generator Drift & Generalization:** Models trained on specific generators (e.g. GPT-3.5 or Stable Diffusion v1.5) may exhibit degraded performance when evaluated on newer, unseen architectures (e.g. Claude 3.5, Midjourney v6, or FLUX.1).
    3. **Post-Processing & Adversarial Perturbations:** Light perturbations such as heavy JPEG compression, audio resampling, paraphrasing, or minor human editing can significantly alter high-frequency feature maps, reducing detector confidence.
    4. **Forensic Use Disclaimer:** Authentica AI is designed as a research, educational, and decision-support tool. It should **never** be used as standalone forensic proof for legal, punitive, or academic misconduct determinations without human-in-the-loop verification.
    """
)
