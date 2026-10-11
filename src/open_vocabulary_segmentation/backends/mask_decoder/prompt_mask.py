from dataclasses import dataclass

from ...array_types import BoolArray
from ...object_prompt import MaskLogits


@dataclass(frozen=True, eq=False)
class PromptMask:
    """
    One decoded mask of one object prompt, before the masks of an image are assembled into a result.

    Attributes
    ----------
    prompt_index : int
        Index of the prompt among the prompts of the image.
    candidate_index : int
        Index of the mask among the masks predicted for the prompt.
    mask : BoolArray
        Mask over the source image, shape (H, W).
    score : float
        Predicted IoU in ``[0, 1]``.
    stability_score : float
        Stability score in ``[0, 1]``.
    mask_logits : MaskLogits
        Low-resolution logits of the mask.
    """

    prompt_index: int
    candidate_index: int
    mask: BoolArray
    score: float
    stability_score: float
    mask_logits: MaskLogits
