"""
Central Model Registry and Metadata catalog for Authentica AI.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    display_name: str
    modality: str
    task: str
    architecture: str
    license: str
    size_mb: int
    training_domain: str
    labels: Dict[int, str]
    input_format: str
    output_format: str
    is_calibrated: bool = False
    notes: Optional[str] = None


MODEL_REGISTRY: Dict[str, ModelMetadata] = {
    "umm-maybe/AI-image-detector": ModelMetadata(
        model_id="umm-maybe/AI-image-detector",
        display_name="ViT AI-Image Detector",
        modality="image",
        task="Binary Image Classification (Artificial vs Human)",
        architecture="Vision Transformer (ViT-Base/16-224)",
        license="MIT",
        size_mb=343,
        training_domain="Human photos vs. Midjourney, Stable Diffusion v1.4/v1.5, DALL-E 2",
        labels={0: "artificial", 1: "human"},
        input_format="RGB 224x224 Normalized Tensor",
        output_format="Softmax Detection Score",
        is_calibrated=False,
        notes="High sensitivity to diffusion textures. Inversion applied so P(AI) = 1.0 - P(human).",
    ),
    "Hello-SimpleAI/chatgpt-detector-roberta": ModelMetadata(
        model_id="Hello-SimpleAI/chatgpt-detector-roberta",
        display_name="RoBERTa ChatGPT Detector",
        modality="text",
        task="Binary Sequence Classification (Human vs ChatGPT)",
        architecture="RoBERTa-base (12-layer transformer)",
        license="Apache 2.0",
        size_mb=499,
        training_domain="Human ChatGPT Comparison Corpus (HC3: QA, Reddit, Wikipedia)",
        labels={0: "Human", 1: "ChatGPT"},
        input_format="Tokenized Text (Max 512 tokens with sliding window chunking)",
        output_format="Softmax Detection Score",
        is_calibrated=False,
        notes="Optimized for English text >= 50 characters.",
    ),
    "garystafford/wav2vec2-deepfake-voice-detector": ModelMetadata(
        model_id="garystafford/wav2vec2-deepfake-voice-detector",
        display_name="Wav2Vec2 Deepfake Voice Detector",
        modality="audio",
        task="Synthetic Speech & Voice Clone Detection",
        architecture="Wav2Vec2 Sequence Classification Head",
        license="Apache 2.0",
        size_mb=378,
        training_domain="ElevenLabs, Uberduck, Amazon Polly vs. LibriSpeech",
        labels={0: "real", 1: "fake"},
        input_format="16,000 Hz Mono PCM Waveform",
        output_format="Softmax Detection Score",
        is_calibrated=False,
        notes="Requires 16kHz resampled mono audio.",
    ),
}


def get_model_metadata(model_id: str) -> Optional[ModelMetadata]:
    """Retrieves metadata record for a given model ID."""
    return MODEL_REGISTRY.get(model_id)


def list_registered_models(modality: Optional[str] = None) -> List[ModelMetadata]:
    """Lists registered models, optionally filtered by modality."""
    if modality:
        return [m for m in MODEL_REGISTRY.values() if m.modality.lower() == modality.lower()]
    return list(MODEL_REGISTRY.values())
