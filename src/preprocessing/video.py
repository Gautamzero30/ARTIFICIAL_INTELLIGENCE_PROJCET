"""
Video preprocessing, keyframe sampling, and audio demuxing for Authentica AI.
"""
import io
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.core.exceptions import CorruptedFileError, ProcessingError, ValidationError


class VideoPreprocessor:
    """
    Extracts representative visual keyframes and demuxes audio tracks from video containers.
    """

    def __init__(
        self,
        sample_frames: int = 8,
        max_duration_sec: int = 60,
    ):
        self.sample_frames = sample_frames
        self.max_duration_sec = max_duration_sec

    def _save_temp_video(self, video_input: Union[str, Path, bytes, BinaryIO]) -> Tuple[str, bool]:
        """
        Saves video to a temporary file if provided as bytes or file stream.
        Returns the file path and a boolean indicating whether it needs cleanup.
        """
        if isinstance(video_input, (str, Path)):
            p = str(video_input)
            if not os.path.exists(p):
                raise CorruptedFileError(f"Video file not found at {p}")
            return p, False

        # Write bytes to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        if isinstance(video_input, bytes):
            temp_file.write(video_input)
        elif hasattr(video_input, "read"):
            data = video_input.read()
            if hasattr(video_input, "seek"):
                video_input.seek(0)
            temp_file.write(data)
        else:
            temp_file.close()
            raise ValidationError(f"Unsupported video input type: {type(video_input)}")

        temp_file.close()
        return temp_file.name, True

    def extract_keyframes(
        self,
        video_input: Union[str, Path, bytes, BinaryIO],
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Extracts N evenly spaced keyframes from the video.
        """
        video_path, is_temp = self._save_temp_video(video_input)

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise CorruptedFileError(f"OpenCV could not open video file.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames <= 0 or fps <= 0:
                cap.release()
                raise CorruptedFileError("Video stream contains 0 frames or invalid FPS.")

            duration_sec = total_frames / fps
            if duration_sec > self.max_duration_sec:
                cap.release()
                raise ValidationError(
                    f"Video duration ({duration_sec:.1f}s) exceeds maximum allowed duration ({self.max_duration_sec}s)."
                )

            # Calculate uniform sampling frame indices
            num_samples = min(self.sample_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

            frames = []
            frame_timestamps = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame_bgr = cap.read()
                if ret and frame_bgr is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    frames.append(pil_img)
                    frame_timestamps.append(round(idx / fps, 2))

            cap.release()

            if not frames:
                raise CorruptedFileError("Failed to decode any valid video frames.")

            metadata = {
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_seconds": round(duration_sec, 2),
                "resolution": f"{width}x{height}",
                "num_keyframes_extracted": len(frames),
                "sample_timestamps": frame_timestamps,
            }
            return frames, metadata

        finally:
            if is_temp and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass

    def extract_audio_track(
        self,
        video_input: Union[str, Path, bytes, BinaryIO],
    ) -> Tuple[Optional[bytes], bool]:
        """
        Extracts the audio track from the video container.
        Returns raw audio bytes and a boolean has_audio flag.
        If the video is silent or has no audio track, returns (None, False).
        """
        video_path, is_temp = self._save_temp_video(video_input)

        try:
            # Attempt extraction via librosa / soundfile / moviepy if available
            import librosa
            try:
                # librosa can read audio directly from many video containers via audioread/ffmpeg
                y, sr = librosa.load(video_path, sr=16000, mono=True)
                if y is not None and len(y) > 0 and not np.all(y == 0):
                    # Write to WAV buffer
                    import wave
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        clipped = np.clip(y, -1.0, 1.0)
                        pcm16 = (clipped * 32767).astype(np.int16)
                        wf.writeframes(pcm16.tobytes())
                    return buf.getvalue(), True
            except Exception:
                pass

            # If librosa failed or no audio stream found
            return None, False

        finally:
            if is_temp and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass
