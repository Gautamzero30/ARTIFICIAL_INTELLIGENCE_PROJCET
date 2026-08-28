"""
Domain-specific exceptions for Authentica AI.
"""

class AuthenticaException(Exception):
    """Base exception for all domain-specific errors in Authentica AI."""
    pass


class ValidationError(AuthenticaException):
    """Raised when an uploaded file or input fails validation checks."""
    pass


class FileSecurityError(ValidationError):
    """Raised when an uploaded file violates security or integrity rules."""
    pass


class CorruptedFileError(ValidationError):
    """Raised when an uploaded file cannot be parsed or is corrupted."""
    pass


class UnsupportedFormatError(ValidationError):
    """Raised when an uploaded file has an unsupported format/MIME type."""
    pass


class InsufficientContentError(ValidationError):
    """Raised when an input text is too short to be analyzed reliably."""
    pass


class ModelLoadError(AuthenticaException):
    """Raised when a machine learning model fails to download or initialize."""
    pass


class InferenceError(AuthenticaException):
    """Raised when an error occurs during model forward pass / inference."""
    pass


class ProcessingError(AuthenticaException):
    """Raised when preprocessing or extraction (e.g. frame/audio demuxing) fails."""
    pass
