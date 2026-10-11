from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from open_vocabulary_detector import ImageSize

from ...array_types import FloatArray, IntArray
from ...exemplars import BoxExemplars
from ...runtime import TorchRuntime


@dataclass(frozen=True, eq=False)
class ExemplarTensors:
    """
    Box exemplars of a group of images in the layout ``Sam3Model`` reads.

    Boxes are normalized by their image size and written as ``(cx, cy, w, h)``; images with fewer boxes than the
    group maximum are padded with boxes labeled ``PADDING_LABEL``, which SAM 3 masks out.

    Attributes
    ----------
    boxes : torch.Tensor
        Normalized CXCYWH boxes on the runtime device and dtype, shape (B, M, 4).
    labels : torch.Tensor
        ``1`` for positive, ``0`` for negative and ``PADDING_LABEL`` for padding boxes, shape (B, M).
    """

    POSITIVE_LABEL: ClassVar[int] = 1
    NEGATIVE_LABEL: ClassVar[int] = 0
    PADDING_LABEL: ClassVar[int] = -10

    boxes: torch.Tensor
    labels: torch.Tensor

    @classmethod
    def build(
        cls, exemplars: Sequence[BoxExemplars], image_sizes: Sequence[ImageSize], runtime: TorchRuntime
    ) -> "ExemplarTensors":
        """
        Normalize and pad the exemplars of a group of images.

        Parameters
        ----------
        exemplars : Sequence[BoxExemplars]
            Exemplars of each image of the group.
        image_sizes : Sequence[ImageSize]
            Size of each image of the group.
        runtime : TorchRuntime
            Device and dtype of the model.

        Raises
        ------
        ValueError
            If the group is empty or the numbers of exemplars and image sizes differ.

        Returns
        -------
        ExemplarTensors
            Padded tensors of the group.
        """
        if not exemplars or len(exemplars) != len(image_sizes):
            raise ValueError(f"expected exemplars for {len(image_sizes)} images. got {len(exemplars)}")
        box_count: int = max(len(entry.xyxy) for entry in exemplars)
        boxes: FloatArray = np.zeros((len(exemplars), box_count, 4), dtype=np.float64)
        labels: IntArray = np.full((len(exemplars), box_count), cls.PADDING_LABEL, dtype=np.int64)
        for index, (entry, image_size) in enumerate(zip(exemplars, image_sizes, strict=True)):
            normalized: FloatArray = entry.xyxy / np.array(
                [image_size.width, image_size.height, image_size.width, image_size.height], dtype=np.float64
            )
            centers: FloatArray = (normalized[:, :2] + normalized[:, 2:]) / 2.0
            extents: FloatArray = normalized[:, 2:] - normalized[:, :2]
            boxes[index, : len(normalized)] = np.concatenate([centers, extents], axis=1)
            labels[index, : len(normalized)] = np.where(entry.is_positive, cls.POSITIVE_LABEL, cls.NEGATIVE_LABEL)
        return cls(
            boxes=runtime.to_model_input(torch.from_numpy(boxes)),
            labels=runtime.to_device(torch.from_numpy(labels)),
        )
