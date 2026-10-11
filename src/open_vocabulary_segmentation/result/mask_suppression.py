import numpy as np

from ..array_types import BoolArray, FloatArray, IntArray
from .mask_overlap import MaskOverlap


class MaskSuppression:
    """
    Greedy non-maximum suppression of binary masks by mask IoU.
    """

    @staticmethod
    def kept_indices(
        masks: BoolArray,
        pixel_xyxy: IntArray,
        scores: FloatArray,
        iou_threshold: float,
        group_ids: IntArray | None = None,
    ) -> IntArray:
        """
        Indices of the masks kept by greedy NMS, most confident first.

        Parameters
        ----------
        masks : BoolArray
            Binary masks, shape (N, H, W).
        pixel_xyxy : IntArray
            Pixel bounds of each mask with exclusive right and bottom edges, shape (N, 4).
        scores : FloatArray
            Score of each mask, shape (N,).
        iou_threshold : float
            Masks overlapping a kept mask with IoU above this value are removed.
        group_ids : IntArray | None, optional
            Group of each mask, shape (N,); masks only suppress masks of their own group. ``None`` suppresses
            across all masks.

        Raises
        ------
        ValueError
            If ``iou_threshold`` is outside ``[0, 1]`` or the shapes disagree.

        Returns
        -------
        IntArray
            Kept indices ordered by descending score, shape (K,).
        """
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1]. got {iou_threshold}")
        mask_count: int = int(masks.shape[0])
        if scores.shape != (mask_count,):
            raise ValueError(f"scores must have shape ({mask_count},). got {scores.shape}")
        if not mask_count:
            return np.zeros((0,), dtype=np.int64)
        overlaps: FloatArray = MaskOverlap.pairwise_iou(masks, pixel_xyxy)
        if group_ids is not None:
            if group_ids.shape != (mask_count,):
                raise ValueError(f"group_ids must have shape ({mask_count},). got {group_ids.shape}")
            is_same_group: BoolArray = group_ids[:, None] == group_ids[None, :]
            overlaps = np.where(is_same_group, overlaps, 0.0)
        is_suppressed: BoolArray = np.zeros(mask_count, dtype=np.bool_)
        kept_indices: list[int] = []
        for index in np.argsort(-scores, kind="stable").tolist():
            if is_suppressed[index]:
                continue
            kept_indices.append(index)
            is_suppressed |= overlaps[index] > iou_threshold
        return np.array(kept_indices, dtype=np.int64)
