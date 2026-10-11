from dataclasses import dataclass

import numpy as np
from open_vocabulary_detector import ImageSize

from ..array_types import Float32Array
from ..options import PromptableBackend


@dataclass(frozen=True, eq=False)
class MaskLogits:
    """
    Low-resolution mask logits predicted for one object, fed back as the mask prompt of a refinement.

    The logits live in the input frame of the model that predicted them, so they are only valid for the same
    checkpoint and the same image; ``ObjectPrompt`` validation rejects them elsewhere.

    Attributes
    ----------
    logits : Float32Array
        Logits over the model input frame, shape (h, w), e.g. (256, 256) for SAM and SAM 2, (288, 288) for SAM 3.
    backend : PromptableBackend
        Family of the model that predicted the logits.
    weights_path : str
        Checkpoint that predicted the logits.
    image_size : ImageSize
        Size of the image the logits were predicted for.

    Raises
    ------
    ValueError
        If the logits are not a two-dimensional finite array.
    """

    logits: Float32Array
    backend: PromptableBackend
    weights_path: str
    image_size: ImageSize

    def __post_init__(self) -> None:
        if self.logits.ndim != 2:
            raise ValueError(f"logits must have shape (h, w). got {self.logits.shape}")
        if not bool(np.isfinite(self.logits).all()):
            raise ValueError("logits must be finite.")
        object.__setattr__(self, "logits", np.ascontiguousarray(self.logits, dtype=np.float32))

    def is_compatible_with(self, backend: PromptableBackend, weights_path: str, image_size: ImageSize) -> bool:
        """
        Check whether the logits can prompt a model on an image.

        Parameters
        ----------
        backend : PromptableBackend
            Family of the model to prompt.
        weights_path : str
            Checkpoint of the model to prompt.
        image_size : ImageSize
            Size of the image to segment.

        Returns
        -------
        bool
            True if the logits come from the same checkpoint and image size.
        """
        return self.backend is backend and self.weights_path == weights_path and self.image_size == image_size
