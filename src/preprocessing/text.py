"""
Text preprocessing, sanitization, and sliding-window chunking for Authentica AI.
"""
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from src.core.exceptions import InsufficientContentError, ValidationError


class TextPreprocessor:
    """
    Handles text sanitization, minimum length enforcement,
    and token-level sliding window chunking for transformer models.
    """

    def __init__(
        self,
        min_characters: int = 50,
        max_tokens: int = 512,
        chunk_overlap: int = 50,
    ):
        self.min_characters = min_characters
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap

    def clean_text(self, raw_text: str) -> str:
        """
        Normalizes Unicode (NFKC), strips excessive whitespace, and removes unprintable characters.
        """
        if not isinstance(raw_text, str):
            raise ValidationError(f"Expected string input, got {type(raw_text).__name__}")

        # Normalize Unicode
        text = unicodedata.normalize("NFKC", raw_text)

        # Replace excessive whitespace / multiple newlines with single spaces/newlines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def validate_text(self, text: str) -> str:
        """
        Validates text length and raises typed exceptions for empty or insufficient inputs.
        """
        cleaned = self.clean_text(text)

        if not cleaned or len(cleaned) == 0:
            raise InsufficientContentError("Submitted text is empty.")

        if len(cleaned) < self.min_characters:
            raise InsufficientContentError(
                f"Submitted text ({len(cleaned)} characters) is too short for reliable AI detection. "
                f"A minimum of {self.min_characters} characters is required."
            )

        return cleaned

    def detect_non_english(self, text: str) -> bool:
        """
        Checks if text contains a significant portion of non-Latin / non-ASCII characters.
        """
        non_ascii_chars = [c for c in text if ord(c) > 127]
        ratio = len(non_ascii_chars) / max(1, len(text))
        return ratio > 0.20

    def chunk_tokens(
        self,
        text: str,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """
        Tokenizes text and splits into overlapping chunks if token length exceeds max_tokens.
        Returns a list of tokenized chunk dicts ready for PyTorch tensors.
        """
        validated_text = self.validate_text(text)

        # Tokenize without truncation first
        encoding = tokenizer(
            validated_text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        total_tokens = len(input_ids)

        # If fits in a single window (accounting for 2 special tokens: <s> and </s>)
        usable_window = self.max_tokens - 2
        if total_tokens <= usable_window:
            chunk = tokenizer(
                validated_text,
                max_length=self.max_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return [chunk]

        # Multi-chunk sliding window
        step = max(1, usable_window - self.chunk_overlap)
        chunks = []
        for start_idx in range(0, total_tokens, step):
            end_idx = min(start_idx + usable_window, total_tokens)
            chunk_token_ids = input_ids[start_idx:end_idx]

            # Decode chunk back to text for consistent tokenizer formatting with special tokens
            chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
            chunk_inputs = tokenizer(
                chunk_text,
                max_length=self.max_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            chunks.append(chunk_inputs)

            if end_idx >= total_tokens:
                break

        return chunks
