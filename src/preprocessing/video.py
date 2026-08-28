"""
Fast Video preprocessing, keyframe sampling, and audio demuxing for Authentica AI.
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
    High-speed video keyframe extraction and audio track demuxing.
    """

    def __init__(
        self,
        sample_frames: int = 6,
        max_duration_sec: int = 60,
    ):
        self.sample_frames = sample_frames
        self.max_duration_sec = max_duration_sec

    def _save_temp_video(self, video_input: Union[str, Path, bytes, BinaryIO]) -> Tuple[str, bool]:
        if isinstance(video_input, (str, Path)):
            p = str(video_input)
            if not os.path.exists(p):
                raise CorruptedFileError(f"Video file not found at {p}")
            return p, False

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
        Fast uniform keyframe extraction using direct frame position seeks.
        """
        video_path, is_temp = self._save_temp_video(video_input)

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise CorruptedFileError("Could not open video stream.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames <= 0 or fps <= 0:
                cap.release()
                raise CorruptedFileError("Video stream contains invalid frame headers.")

            duration_sec = total_frames / fps
            if duration_sec > self.max_duration_sec:
                cap.release()
                raise ValidationError(
                    f"Video duration ({duration_sec:.1f}s) exceeds maximum allowed duration ({self.max_duration_sec}s)."
                )

            # Sample 4-6 frames uniformly
            num_samples = min(self.sample_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

            frames = []
            frame_timestamps = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame_bgr = cap.read()
                if ret and frame_bgr is not None:
                    # Resize to max 512px height for faster processing
                    h, w = frame_bgr.shape[:2]
                    if max(h, w) > 512:
                        scale = 512.0 / max(h, w)
                        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    frames.append(pil_img)
                    frame_timestamps.append(round(float(idx / fps), 2))

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
        Fast audio demuxing with 10-second duration cap for instant latency.
        """
        video_path, is_temp = self._save_temp_video(video_input)

        try:
            import librosa
            try:
                # Load first 10 seconds of audio with duration cap for lightning speed
                y, sr = librosa.load(video_path, sr=16000, mono=True, duration=10.0)
                if y is not None and len(y) > 1600:  # at least 0.1s
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

            return None, False

        finally:
            if is_temp and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass
