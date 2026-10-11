from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from open_vocabulary_detector import ImageSize
from torch.nn import functional

from ..array_types import BoolArray, Float32Array, FloatArray


@dataclass(frozen=True)
class ModelFrame:
    """
    Placement of a source image inside the square input of a promptable model.

    The image is resized to ``resized_size`` and placed at the top-left corner of a ``input_size`` square, padded
    below and to the right: SAM keeps the aspect ratio and pads, SAM 2 and SAM 3 stretch the image over the whole
    square. Prompts are mapped from image pixels into this frame.

    Attributes
    ----------
    image_size : ImageSize
        Size of the source image.
    resized_size : ImageSize
        Size of the resized image inside the model input.
    input_size : int
        Side of the square model input in pixels.

    Raises
    ------
    ValueError
        If the resized image does not fit into the model input.
    """

    MASK_PROMPT_LOGIT: ClassVar[float] = 10.0

    image_size: ImageSize
    resized_size: ImageSize
    input_size: int

    def __post_init__(self) -> None:
        if self.resized_size.width > self.input_size or self.resized_size.height > self.input_size:
            raise ValueError(
                f"resized size ({self.resized_size.width}, {self.resized_size.height}) must fit into the "
                + f"{self.input_size} model input."
            )

    @property
    def scale_xy(self) -> FloatArray:
        """
        Factors mapping image pixels to model input pixels.

        Returns
        -------
        FloatArray
            ``(scale_x, scale_y)``, shape (2,).
        """
        return np.array(
            [
                self.resized_size.width / self.image_size.width,
                self.resized_size.height / self.image_size.height,
            ],
            dtype=np.float64,
        )

    def to_model_xy(self, point_xy: FloatArray) -> FloatArray:
        """
        Map points from image pixels to model input pixels.

        Parameters
        ----------
        point_xy : FloatArray
            ``(x, y)`` image pixels, shape (..., 2).

        Returns
        -------
        FloatArray
            ``(x, y)`` model input pixels, same shape.
        """
        return point_xy * self.scale_xy

    def to_model_xyxy(self, xyxy: FloatArray) -> FloatArray:
        """
        Map XYXY boxes from image pixels to model input pixels.

        Parameters
        ----------
        xyxy : FloatArray
            Boxes in image pixels, shape (..., 4).

        Returns
        -------
        FloatArray
            Boxes in model input pixels, same shape.
        """
        return xyxy * np.tile(self.scale_xy, 2)

    def to_mask_logits(self, mask: BoolArray, mask_input_size: int) -> Float32Array:
        """
        Turn a binary mask of the image into low-resolution mask logits in the model input frame.

        The mask is resized like the image, padded like the image and downsampled to the mask prompt resolution;
        inside pixels map to ``+MASK_PROMPT_LOGIT``, outside pixels to ``-MASK_PROMPT_LOGIT``, with a smooth
        transition along the boundary.

        Parameters
        ----------
        mask : BoolArray
            Mask of the source image, shape (H, W).
        mask_input_size : int
            Side of the model's mask prompt, e.g. 256 for SAM.

        Raises
        ------
        ValueError
            If the mask does not cover the source image.

        Returns
        -------
        Float32Array
            Mask logits, shape (mask_input_size, mask_input_size).
        """
        if mask.shape != (self.image_size.height, self.image_size.width):
            raise ValueError(
                f"mask must have shape ({self.image_size.height}, {self.image_size.width}). got {mask.shape}"
            )
        values: torch.Tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None]
        resized: torch.Tensor = functional.interpolate(
            values,
            size=(self.resized_size.height, self.resized_size.width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        padded: torch.Tensor = functional.pad(
            resized, (0, self.input_size - self.resized_size.width, 0, self.input_size - self.resized_size.height)
        )
        downsampled: torch.Tensor = functional.interpolate(
            padded, size=(mask_input_size, mask_input_size), mode="bilinear", align_corners=False, antialias=True
        )
        logits: torch.Tensor = (downsampled[0, 0].clamp(0.0, 1.0) * 2.0 - 1.0) * self.MASK_PROMPT_LOGIT
        return logits.numpy().astype(np.float32)
