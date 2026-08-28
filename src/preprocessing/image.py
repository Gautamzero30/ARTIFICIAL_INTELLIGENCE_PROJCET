"""
Image preprocessing and validation pipeline for Authentica AI.
"""
import io
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps

from src.core.exceptions import CorruptedFileError, ValidationError

# Standard ImageNet normalization parameters used by ViT
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImagePreprocessor:
    """
    Handles safe image decoding, color conversion, EXIF correction,
    resizing, and tensor normalization for Vision Transformer models.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        mean: Optional[list] = None,
        std: Optional[list] = None,
    ):
        self.target_size = target_size
        self.mean = torch.tensor(mean or IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(std or IMAGENET_STD).view(3, 1, 1)

    def load_image(
        self, image_input: Union[str, Path, bytes, BinaryIO, Image.Image]
    ) -> Image.Image:
        """
        Safely loads an image from various input types and ensures 3-channel RGB format.
        """
        try:
            if isinstance(image_input, Image.Image):
                img = image_input
            elif isinstance(image_input, (str, Path)):
                img = Image.open(image_input)
            elif isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input))
            elif hasattr(image_input, "read"):
                data = image_input.read()
                if hasattr(image_input, "seek"):
                    image_input.seek(0)
                img = Image.open(io.BytesIO(data))
            else:
                raise ValidationError(f"Unsupported image input type: {type(image_input)}")

            # Auto-correct orientation based on EXIF metadata
            img = ImageOps.exif_transpose(img)

            # Ensure image is in 3-channel RGB format
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Force load image data to catch truncation/corruption early
            img.load()
            return img

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise CorruptedFileError(f"Failed to decode or parse image: {e}")

    def preprocess(
        self,
        image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
    ) -> torch.Tensor:
        """
        Converts raw image input into a normalized PyTorch tensor [1, 3, H, W].
        """
        pil_img = self.load_image(image_input)

        # High-quality bicubic resampling for ViT
        resized_img = pil_img.resize(self.target_size, Image.Resampling.BICUBIC)

        # Convert to numpy array [H, W, 3] in range [0, 1]
        np_img = np.array(resized_img, dtype=np.float32) / 255.0

        # Permute to channels-first: [3, H, W]
        tensor = torch.from_numpy(np_img).permute(2, 0, 1)

        # Apply ImageNet normalization: (x - mean) / std
        normalized_tensor = (tensor - self.mean) / self.std

        # Add batch dimension: [1, 3, H, W]
        return normalized_tensor.unsqueeze(0)
