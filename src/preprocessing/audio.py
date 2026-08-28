"""
Audio preprocessing, resampling, and chunking pipeline for Authentica AI.
"""
import io
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.core.exceptions import CorruptedFileError, ValidationError


class AudioPreprocessor:
    """
    Handles safe audio loading, stereo-to-mono conversion,
    16kHz resampling, amplitude normalization, and chunking.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        chunk_duration_sec: float = 4.0,
    ):
        self.target_sr = target_sr
        self.chunk_duration_sec = chunk_duration_sec
        self.chunk_samples = int(target_sr * chunk_duration_sec)

    def load_audio(
        self,
        audio_input: Union[str, Path, bytes, BinaryIO],
    ) -> Tuple[np.ndarray, int]:
        """
        Safely loads audio waveform and original sample rate.
        Converts stereo/multichannel to mono.
        """
        # Strategy 1: Standard library wave module for WAV files
        try:
            import wave
            wf = None
            if isinstance(audio_input, (str, Path)):
                wf = wave.open(str(audio_input), "rb")
            elif isinstance(audio_input, bytes):
                wf = wave.open(io.BytesIO(audio_input), "rb")
            elif hasattr(audio_input, "read"):
                raw_bytes = audio_input.read()
                if hasattr(audio_input, "seek"):
                    audio_input.seek(0)
                wf = wave.open(io.BytesIO(raw_bytes), "rb")

            if wf is not None:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
                wf.close()

                if sampwidth == 2:
                    data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    data = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

                if n_channels > 1:
                    data = data.reshape(-1, n_channels).mean(axis=1)

                if len(data) == 0:
                    raise CorruptedFileError("Audio file contains no decodable audio samples.")

                return data, sr
        except Exception:
            pass

        # Strategy 2: librosa fallback for MP3 / FLAC / OGG
        try:
            import librosa
            if isinstance(audio_input, (str, Path)):
                data, sr = librosa.load(str(audio_input), sr=self.target_sr, mono=True)
                return data, sr
            elif isinstance(audio_input, bytes):
                data, sr = librosa.load(io.BytesIO(audio_input), sr=self.target_sr, mono=True)
                return data, sr
            elif hasattr(audio_input, "read"):
                raw = audio_input.read()
                if hasattr(audio_input, "seek"):
                    audio_input.seek(0)
                data, sr = librosa.load(io.BytesIO(raw), sr=self.target_sr, mono=True)
                return data, sr
        except Exception as lib_e:
            raise CorruptedFileError(f"Failed to decode or parse audio file: {lib_e}")

        raise CorruptedFileError("Could not decode audio with any supported audio engine.")

    def resample(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resamples waveform to target sampling rate (16,000 Hz).
        """
        if orig_sr == self.target_sr:
            return waveform

        import librosa
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=self.target_sr)

    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """
        Applies peak amplitude normalization to prevent clipping and scale disparities.
        """
        max_val = np.max(np.abs(waveform))
        if max_val > 1e-6:
            return waveform / max_val
        return waveform

    def chunk_waveform(self, waveform: np.ndarray) -> List[np.ndarray]:
        """
        Splits waveform into fixed 4-second chunks (64,000 samples).
        If audio is shorter than 4 seconds, zero-pads to 4 seconds.
        """
        total_samples = len(waveform)

        # If audio is shorter than minimum chunk duration, pad with zeros
        if total_samples < self.chunk_samples:
            padded = np.pad(waveform, (0, self.chunk_samples - total_samples), mode="constant")
            return [padded]

        # Split into consecutive non-overlapping chunks
        chunks = []
        for start_idx in range(0, total_samples, self.chunk_samples):
            end_idx = start_idx + self.chunk_samples
            chunk = waveform[start_idx:end_idx]

            # Pad final chunk if incomplete
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode="constant")
            chunks.append(chunk)

        return chunks

    def preprocess(
        self,
        audio_input: Union[str, Path, bytes, BinaryIO],
    ) -> Tuple[List[torch.Tensor], Dict[str, Union[int, float]]]:
        """
        Complete preprocessing pipeline: loads, converts to mono, resamples to 16kHz,
        normalizes, and converts chunks into PyTorch tensors.
        """
        raw_waveform, orig_sr = self.load_audio(audio_input)
        duration_sec = len(raw_waveform) / float(orig_sr)

        resampled = self.resample(raw_waveform, orig_sr)
        normalized = self.normalize(resampled)
        chunks = self.chunk_waveform(normalized)

        tensor_chunks = [torch.from_numpy(c).float().unsqueeze(0) for c in chunks]

        metadata = {
            "original_sample_rate": orig_sr,
            "target_sample_rate": self.target_sr,
            "duration_seconds": round(duration_sec, 2),
            "num_chunks": len(tensor_chunks),
        }
        return tensor_chunks, metadata
