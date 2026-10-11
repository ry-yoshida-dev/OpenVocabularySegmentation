from dataclasses import dataclass

import torch
from open_vocabulary_detector import ImageSize

from ..options import PromptableBackend
from .model_frame import ModelFrame


@dataclass(frozen=True, eq=False)
class EncodedImage:
    """
    Image embedding computed once by a promptable segmenter and reused for any number of prompts.

    Encoding is the expensive step of SAM-family models; prompting an encoded image only runs the light prompt
    encoder and mask decoder, which makes click-by-click refinement interactive. An encoding is only valid for the
    segmenter (checkpoint) that produced it.

    Attributes
    ----------
    features : tuple[torch.Tensor, ...]
        Image features on the model device, each with batch size 1; one tensor for SAM, the multi-scale features
        for SAM 2 and SAM 3.
    frame : ModelFrame
        Placement of the image inside the model input.
    backend : PromptableBackend
        Family of the segmenter that encoded the image.
    weights_path : str
        Checkpoint of the segmenter that encoded the image.

    Raises
    ------
    ValueError
        If there is no feature tensor or a tensor has a batch size other than 1.
    """

    features: tuple[torch.Tensor, ...]
    frame: ModelFrame
    backend: PromptableBackend
    weights_path: str

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("an encoded image needs at least one feature tensor.")
        if any(feature.shape[0] != 1 for feature in self.features):
            raise ValueError(f"features must have batch size 1. got {[tuple(f.shape) for f in self.features]}")

    @property
    def image_size(self) -> ImageSize:
        """
        Size of the encoded source image.

        Returns
        -------
        ImageSize
            Width and height of the source image.
        """
        return self.frame.image_size
