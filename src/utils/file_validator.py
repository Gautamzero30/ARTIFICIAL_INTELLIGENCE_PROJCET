"""
File security, MIME checking, and payload validation utilities for Authentica AI.
"""
import io
import os
import re
from pathlib import Path
from typing import BinaryIO, Optional, Union

from src.core.config import SecurityConfig
from src.core.exceptions import (
    CorruptedFileError,
    FileSecurityError,
    UnsupportedFormatError,
    ValidationError,
)

# Magic bytes signatures for trusted formats
MAGIC_SIGNATURES = {
    # Images
    "jpeg": [b"\xFF\xD8\xFF"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF", b"WEBP"],  # RIFF....WEBP
    "bmp": [b"BM"],
    # Audio
    "wav": [b"RIFF", b"WAVE"],  # RIFF....WAVE
    "mp3": [b"\xFF\xFB", b"\xFF\xF3", b"\xFF\xF2", b"ID3"],
    "flac": [b"fLaC"],
    "ogg": [b"OggS"],
    # Video
    "mp4": [b"ftyp", b"moov"],   # ftyp / moov
    "webm": [b"\x1A\x45\xDF\xA3"], # EBML header
    "avi": [b"RIFF", b"AVI "],
    "mkv": [b"\x1A\x45\xDF\xA3"],
}


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes user-provided filename to prevent path traversal and shell injection.
    """
    if not filename:
        return "unnamed_file"
    # Strip directory paths
    base = os.path.basename(filename)
    # Remove dangerous characters
    sanitized = re.sub(r"[^\w\.\-_]", "_", base)
    # Prevent leading dots or hidden files
    sanitized = sanitized.lstrip(".")
    return sanitized or "unnamed_file"


class FileValidator:
    """
    Validates uploaded files for size, extension, MIME type, and magic bytes.
    """

    def __init__(self, security_cfg: Optional[SecurityConfig] = None):
        self.security_cfg = security_cfg or SecurityConfig()

    def validate_file(
        self,
        file_obj: Union[BinaryIO, bytes, io.BytesIO],
        filename: str,
        modality: str,
    ) -> bytes:
        """
        Validates file stream/bytes against size, extension, and magic bytes.
        Returns the raw validated bytes.
        """
        modality = modality.lower()
        if modality not in ["image", "audio", "video", "text"]:
            raise ValidationError(f"Unknown modality '{modality}'. Expected image, audio, video, or text.")

        # Extract bytes
        if isinstance(file_obj, bytes):
            data = file_obj
        elif hasattr(file_obj, "read"):
            data = file_obj.read()
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
        else:
            raise ValidationError("Invalid file object provided.")

        # 1. Check for empty file
        if not data or len(data) == 0:
            raise CorruptedFileError(f"Uploaded file '{filename}' is empty.")

        # 2. File size validation
        size_mb = len(data) / (1024 * 1024)
        max_size = {
            "image": self.security_cfg.max_image_size_mb,
            "audio": self.security_cfg.max_audio_size_mb,
            "video": self.security_cfg.max_video_size_mb,
            "text": 5, # 5MB max for text
        }.get(modality, 10)

        if size_mb > max_size:
            raise FileSecurityError(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit for {modality} ({max_size} MB)."
            )

        # 3. Extension check
        ext = Path(filename).suffix.lower()
        allowed_exts = self.security_cfg.allowed_extensions.get(modality, [])
        if ext not in allowed_exts:
            raise UnsupportedFormatError(
                f"Extension '{ext}' is not supported for {modality}. Allowed: {', '.join(allowed_exts)}"
            )

        # 4. Deep magic byte validation (for binary files)
        if modality != "text":
            self._verify_magic_bytes(data, ext, modality)

        return data

    def _verify_magic_bytes(self, data: bytes, ext: str, modality: str) -> None:
        """
        Inspects header bytes to detect extension spoofing.
        """
        header = data[:64]
        ext_clean = ext.lstrip(".").lower()
        
        # JPEG
        if ext_clean in ["jpg", "jpeg"]:
            if not header.startswith(b"\xFF\xD8\xFF"):
                raise CorruptedFileError("File header does not match JPEG magic signature (possible format spoofing).")
        # PNG
        elif ext_clean == "png":
            if not header.startswith(b"\x89PNG\r\n\x1a\n"):
                raise CorruptedFileError("File header does not match PNG magic signature.")
        # WEBP / WAV / AVI (RIFF containers)
        elif ext_clean == "webp":
            if not (header.startswith(b"RIFF") and b"WEBP" in header[:16]):
                raise CorruptedFileError("File header does not match WebP RIFF signature.")
        elif ext_clean == "wav":
            if not (header.startswith(b"RIFF") and b"WAVE" in header[:16]):
                raise CorruptedFileError("File header does not match WAV RIFF signature.")
        # MP3
        elif ext_clean == "mp3":
            is_id3 = header.startswith(b"ID3")
            is_raw_mp3 = any(header.startswith(sig) for sig in [b"\xFF\xFB", b"\xFF\xF3", b"\xFF\xF2"])
            if not (is_id3 or is_raw_mp3):
                raise CorruptedFileError("File header does not match MP3 sync word or ID3 tag.")
        # FLAC
        elif ext_clean == "flac":
            if not header.startswith(b"fLaC"):
                raise CorruptedFileError("File header does not match FLAC magic signature.")
        # OGG
        elif ext_clean == "ogg":
            if not header.startswith(b"OggS"):
                raise CorruptedFileError("File header does not match OGG magic signature.")
        # MP4 / MOV
        elif ext_clean in ["mp4", "mov", "m4a"]:
            if not (b"ftyp" in header[4:16] or b"moov" in header[:32]):
                raise CorruptedFileError(f"File header does not match {ext_clean.upper()} container signature.")
        # WEBM / MKV
        elif ext_clean in ["webm", "mkv"]:
            if not header.startswith(b"\x1A\x45\xDF\xA3"):
                raise CorruptedFileError(f"File header does not match {ext_clean.upper()} EBML signature.")
