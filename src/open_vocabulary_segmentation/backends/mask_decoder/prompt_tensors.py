from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from image_container import BinaryImage

from ...array_types import Float32Array, FloatArray, IntArray
from ...object_prompt import MaskLogits, MaskPrompt, ObjectPrompt
from ...promptable import ModelFrame
from ...runtime import TorchRuntime
from .point_label import PointLabel
from .prompt_signature import PromptSignature


@dataclass(frozen=True, eq=False)
class PromptTensors:
    """
    Object prompts of one signature in the model input frame, in the layout SAM-family decoders read.

    Attributes
    ----------
    points : torch.Tensor | None
        Point coordinates, shape (1, N, P, 2); a ``NOT_A_POINT`` placeholder when the prompts have neither points
        nor a box; ``None`` for box prompts without points.
    labels : torch.Tensor | None
        ``PointLabel`` of every point, shape (1, N, P); present exactly when ``points`` is.
    boxes : torch.Tensor | None
        XYXY boxes, shape (1, N, 4); ``None`` without boxes.
    masks : torch.Tensor | None
        Mask logits, shape (N, 1, h, w); ``None`` without masks.
    """

    points: torch.Tensor | None
    labels: torch.Tensor | None
    boxes: torch.Tensor | None
    masks: torch.Tensor | None

    @classmethod
    def build(
        cls,
        prompts: Sequence[ObjectPrompt],
        frame: ModelFrame,
        mask_input_size: int,
        runtime: TorchRuntime,
    ) -> "PromptTensors":
        """
        Map prompts of one signature into the model input frame.

        Parameters
        ----------
        prompts : Sequence[ObjectPrompt]
            Prompts sharing one ``PromptSignature``, validated for the image.
        frame : ModelFrame
            Placement of the image inside the model input.
        mask_input_size : int
            Side of the model's mask prompt.
        runtime : TorchRuntime
            Device and dtype of the model.

        Raises
        ------
        ValueError
            If there is no prompt, the prompts have different signatures, or mask logits have another resolution
            than the model's mask prompt.

        Returns
        -------
        PromptTensors
            Tensors on the runtime device.
        """
        if not prompts:
            raise ValueError("at least one prompt is needed.")
        signature: PromptSignature = PromptSignature.of(prompts[0])
        if any(PromptSignature.of(prompt) != signature for prompt in prompts):
            raise ValueError("prompts decoded together must share one signature.")
        points: torch.Tensor | None = None
        labels: torch.Tensor | None = None
        if signature.point_count:
            point_xy: FloatArray = np.stack([frame.to_model_xy(prompt.point_xy) for prompt in prompts])
            point_labels: IntArray = np.stack(
                [
                    np.where(prompt.is_point_positive, PointLabel.POSITIVE, PointLabel.NEGATIVE).astype(np.int64)
                    for prompt in prompts
                ]
            )
            points = runtime.to_model_input(torch.from_numpy(point_xy)[None])
            labels = runtime.to_device(torch.from_numpy(point_labels)[None])
        elif not signature.is_box_given:
            points = runtime.to_model_input(torch.zeros(1, len(prompts), 1, 2))
            labels = runtime.to_device(torch.full((1, len(prompts), 1), int(PointLabel.NOT_A_POINT), dtype=torch.int64))
        boxes: torch.Tensor | None = None
        if signature.is_box_given:
            xyxy: FloatArray = np.stack([frame.to_model_xyxy(cls._xyxy_of(prompt)) for prompt in prompts]).astype(
                np.float64
            )
            boxes = runtime.to_model_input(torch.from_numpy(xyxy)[None])
        masks: torch.Tensor | None = None
        if signature.is_mask_given:
            mask_logits: Float32Array = np.stack(
                [cls._mask_logits_of(cls._mask_of(prompt), frame, mask_input_size) for prompt in prompts]
            )
            masks = runtime.to_model_input(torch.from_numpy(mask_logits)[:, None])
        return cls(points=points, labels=labels, boxes=boxes, masks=masks)

    @staticmethod
    def _xyxy_of(prompt: ObjectPrompt) -> FloatArray:
        xyxy: FloatArray | None = prompt.xyxy
        if xyxy is None:
            raise ValueError("every prompt of a box signature needs a box.")
        return xyxy

    @staticmethod
    def _mask_of(prompt: ObjectPrompt) -> MaskPrompt:
        if prompt.mask is None:
            raise ValueError("every prompt of a mask signature needs a mask.")
        return prompt.mask

    @staticmethod
    def _mask_logits_of(mask: MaskPrompt, frame: ModelFrame, mask_input_size: int) -> Float32Array:
        match mask:
            case MaskLogits(logits=logits):
                if logits.shape != (mask_input_size, mask_input_size):
                    raise ValueError(
                        f"mask logits must have shape ({mask_input_size}, {mask_input_size}). got {logits.shape}"
                    )
                return logits
            case BinaryImage(value=value):
                return frame.to_mask_logits(np.asarray(value, dtype=np.bool_), mask_input_size)
