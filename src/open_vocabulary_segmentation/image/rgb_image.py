from typing import ClassVar

import numpy as np
from PIL import Image

from ..array_types import FloatArray
from .image_input import ImageInput


class RgbImage:
    """
    Converts the images given to segmenters and trackers into the RGB PIL images the models read.

    PIL clips high bit depth grayscale images (16-bit integer, 32-bit integer and float modes) to 255 when converting
    them to RGB, so they are first stretched from their minimum to their maximum value onto 8 bits.
    """

    HIGH_BIT_DEPTH_MODES: ClassVar[frozenset[str]] = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N", "F"})
    MAX_8_BIT_VALUE: ClassVar[float] = 255.0

    @classmethod
    def from_input(cls, image: ImageInput) -> Image.Image:
        """
        Convert a PIL image of any mode or an ``image_container`` container of any channel order to RGB.

        Parameters
        ----------
        image : ImageInput
            Input image.

        Returns
        -------
        Image.Image
            RGB PIL image; ``image`` itself if it already is one.
        """
        pil_image: Image.Image = image if isinstance(image, Image.Image) else image.to_PIL()
        if pil_image.mode == "RGB":
            return pil_image
        if pil_image.mode in cls.HIGH_BIT_DEPTH_MODES:
            return cls._stretch_to_8_bit(pil_image).convert("RGB")
        return pil_image.convert("RGB")

    @classmethod
    def _stretch_to_8_bit(cls, image: Image.Image) -> Image.Image:
        values: FloatArray = np.asarray(image, dtype=np.float64)
        finite_values: FloatArray = values[np.isfinite(values)]
        if not finite_values.size:
            return Image.fromarray(np.zeros(values.shape, dtype=np.uint8))
        minimum: float = float(finite_values.min())
        value_range: float = float(finite_values.max()) - minimum
        scaled: FloatArray = (
            np.zeros_like(values)
            if value_range == 0.0
            else (np.nan_to_num(values, nan=minimum) - minimum) * (cls.MAX_8_BIT_VALUE / value_range)
        )
        return Image.fromarray(np.clip(np.rint(scaled), 0.0, cls.MAX_8_BIT_VALUE).astype(np.uint8))
