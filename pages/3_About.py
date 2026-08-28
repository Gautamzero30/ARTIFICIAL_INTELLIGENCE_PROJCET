"""
Authentica AI — Enterprise Documentation, Model Registry, and Responsible AI Governance.
"""
import streamlit as st
from src.utils.model_registry import list_registered_models

st.set_page_config(
    page_title="Architecture & Governance — Authentica AI",
    page_icon="ℹ️",
    layout="wide",
)

st.title("ℹ️ Architecture & Responsible AI Governance")
st.markdown("System specifications, candidate model registry, and ethical boundaries.")

# System Architecture Flow
st.markdown("### 🏗️ Enterprise Multimodal Architecture")
st.markdown(
    """
    ```text
                               AUTHENTICA AI MULTIMODAL PIPELINE
                                              │
         ┌────────────────────────┬───────────┴───────────┬────────────────────────┐
         ▼                        ▼                       ▼                        ▼
     🖼️ IMAGE                  📝 TEXT                 🎵 AUDIO                 🎬 VIDEO
  Vision Transformer          RoBERTa-base            Wav2Vec2                 Decoupled Late
(umm-maybe/AI-image)        (HC3 QA Corpus)        (Deepfake Voice)                Fusion
         │                        │                       │                        │
    RGB & EXIF                Unicode & Token         16kHz Mono              ┌────┴────┐
   Normalization                  Chunker             Resampling              ▼         ▼
         │                        │                       │                 Frames    Audio
         │                        │                       │                 (ViT)   (Wav2Vec2)
         │                        │                       │                   │         │
         │                        │                       │                   └────┬────┘
         │                        │                       │                        ▼
         │                        │                       │                 Weighted Late
         │                        │                       │                    Fusion
         └────────────────────────┼───────────────────────┼────────────────────────┘
                                  │
                                  ▼
                      Decision Classification Engine
                      • S_AI >= 0.65  ──► LIKELY AI-GENERATED
                      • S_AI <= 0.35  ──► LIKELY HUMAN-CREATED
                      • 0.35 < S < 0.65 ─► UNCERTAIN
                                  │
                                  ▼
                    Forensic Verdicts & Disclaimers
    ```
    """
)

# Model Registry
st.markdown("---")
st.markdown("### 🤖 Candidate Model Registry")
models = list_registered_models()
for m in models:
    with st.expander(f"🔹 {m.display_name} (`{m.model_id}`)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Modality:** `{m.modality.capitalize()}`")
            st.write(f"**Architecture:** `{m.architecture}`")
            st.write(f"**Task Domain:** `{m.task}`")
            st.write(f"**Model Size:** `~{m.size_mb} MB`")
            st.write(f"**License:** `{m.license}`")
        with c2:
            st.write(f"**Training Domain:** `{m.training_domain}`")
            st.write(f"**Input Specification:** `{m.input_format}`")
            st.write(f"**Output Format:** `{m.output_format}`")
            if m.notes:
                st.info(f"📌 {m.notes}")

# Responsible AI & Limitations
st.markdown("---")
st.markdown("### ⚖️ Responsible AI Principles & Known Limitations")
st.markdown(
    """
    1. **Probabilistic Confidence:** Authentica AI computes statistical likelihood estimates based on observed high-frequency residuals, token perplexity, and acoustic harmonics. It is not an ontological ground-truth detector.
    2. **Adversarial Post-Processing:** Content modified with compression artifacts, deliberate paraphrasing, low-pass audio filters, or hybrid human-in-the-loop editing will alter feature maps and moderate detector confidence.
    3. **Generator Generalization:** Detectors demonstrate high accuracy on generator families aligned with their training corpora. Performance degradation may occur on completely unrepresented zero-shot foundation models.
    4. **Forensic Use Policy:** This tool is engineered for trust & safety analysis, research, and content moderation screening. It must not be deployed as the sole basis for punitive academic or legal determinations.
    """
)
