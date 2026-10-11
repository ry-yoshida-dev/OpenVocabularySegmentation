from dataclasses import dataclass

import torch
from open_vocabulary_detector import ImageSize

from ...promptable import ModelFrame


@dataclass(frozen=True, eq=False)
class PreprocessedImages:
    """
    Images of a mini-batch resized and normalized for the image encoder.

    Attributes
    ----------
    pixel_values : torch.Tensor
        Model input on the CPU, shape (B, C, S, S).
    frames : tuple[ModelFrame, ...]
        Placement of each image inside the model input, length B.

    Raises
    ------
    ValueError
        If the number of frames differs from the batch size.
    """

    pixel_values: torch.Tensor
    frames: tuple[ModelFrame, ...]

    def __post_init__(self) -> None:
        if self.pixel_values.shape[0] != len(self.frames):
            raise ValueError(f"expected {self.pixel_values.shape[0]} frames. got {len(self.frames)}")

    @classmethod
    def stretched(cls, pixel_values: torch.Tensor, original_sizes: torch.Tensor) -> "PreprocessedImages":
        """
        Describe images stretched over the whole square model input, as SAM 2 and SAM 3 preprocess them.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Model input, shape (B, C, S, S).
        original_sizes : torch.Tensor
            ``(height, width)`` of every source image, shape (B, 2).

        Returns
        -------
        PreprocessedImages
            Pixel values with one full-square frame per image.
        """
        input_size: int = int(pixel_values.shape[-1])
        return cls(
            pixel_values=pixel_values,
            frames=tuple(
                ModelFrame(
                    image_size=ImageSize(width=int(original_sizes[index, 1]), height=int(original_sizes[index, 0])),
                    resized_size=ImageSize(width=input_size, height=input_size),
                    input_size=input_size,
                )
                for index in range(int(original_sizes.shape[0]))
            ),
        )
