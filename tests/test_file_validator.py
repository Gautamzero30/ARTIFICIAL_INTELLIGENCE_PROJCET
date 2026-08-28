"""
Unit tests for file security, magic bytes, and validation.
"""
import io
import pytest

from src.core.config import SecurityConfig
from src.core.exceptions import (
    CorruptedFileError,
    FileSecurityError,
    UnsupportedFormatError,
    ValidationError,
)
from src.utils.file_validator import FileValidator, sanitize_filename


def test_sanitize_filename():
    """Verify filename sanitization prevents path traversal and dangerous characters."""
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("test image!@#$%.png") == "test_image_____.png"
    assert sanitize_filename(".hidden_file.jpg") == "hidden_file.jpg"
    assert sanitize_filename("") == "unnamed_file"


def test_empty_file_rejected():
    """Verify empty file raises CorruptedFileError."""
    validator = FileValidator()
    with pytest.raises(CorruptedFileError):
        validator.validate_file(b"", "empty.jpg", "image")


def test_unsupported_extension_rejected():
    """Verify unsupported file extension raises UnsupportedFormatError."""
    validator = FileValidator()
    with pytest.raises(UnsupportedFormatError):
        validator.validate_file(b"some content", "test.exe", "image")


def test_file_size_limit_exceeded():
    """Verify oversized files trigger FileSecurityError."""
    cfg = SecurityConfig(max_image_size_mb=1)
    validator = FileValidator(cfg)
    oversized_data = b"\xFF\xD8\xFF" + b"0" * (2 * 1024 * 1024)  # 2MB
    with pytest.raises(FileSecurityError):
        validator.validate_file(oversized_data, "large.jpg", "image")


def test_valid_jpeg_magic_bytes():
    """Verify valid JPEG passes magic byte validation."""
    validator = FileValidator()
    valid_jpeg = b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"\x00" * 100
    res = validator.validate_file(valid_jpeg, "photo.jpg", "image")
    assert res == valid_jpeg


def test_spoofed_jpeg_rejected():
    """Verify JPEG extension with non-JPEG magic bytes is rejected."""
    validator = FileValidator()
    fake_jpeg = b"THIS IS NOT A REAL JPEG"
    with pytest.raises(CorruptedFileError):
        validator.validate_file(fake_jpeg, "fake.jpg", "image")


def test_valid_wav_magic_bytes():
    """Verify valid WAV audio passes magic byte validation."""
    validator = FileValidator()
    valid_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 50
    res = validator.validate_file(valid_wav, "speech.wav", "audio")
    assert res == valid_wav
