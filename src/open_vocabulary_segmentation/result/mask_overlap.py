import numpy as np

from ..array_types import BoolArray, FloatArray, IntArray


class MaskOverlap:
    """
    Pairwise intersection over union of binary masks.

    A mask lies inside its bounding box, so two masks can only intersect inside the intersection of their boxes;
    pixels are counted there only, and pairs with disjoint boxes are never compared. This keeps the cost
    proportional to the overlapping area instead of the image size.
    """

    @staticmethod
    def pairwise_iou(masks: BoolArray, pixel_xyxy: IntArray) -> FloatArray:
        """
        Intersection over union of every pair of masks.

        Parameters
        ----------
        masks : BoolArray
            Binary masks, shape (N, H, W).
        pixel_xyxy : IntArray
            Bounding box of each mask as pixel bounds with exclusive right and bottom edges, shape (N, 4).

        Raises
        ------
        ValueError
            If the masks are not three-dimensional or the box count differs from the mask count.

        Returns
        -------
        FloatArray
            Symmetric IoU matrix with ones on the diagonal for non-empty masks, shape (N, N).
        """
        if masks.ndim != 3:
            raise ValueError(f"masks must have shape (N, H, W). got {masks.shape}")
        mask_count: int = int(masks.shape[0])
        if pixel_xyxy.shape != (mask_count, 4):
            raise ValueError(f"pixel_xyxy must have shape ({mask_count}, 4). got {pixel_xyxy.shape}")
        areas: IntArray = masks.sum(axis=(1, 2)).astype(np.int64)
        overlaps: FloatArray = np.zeros((mask_count, mask_count), dtype=np.float64)
        for first in range(mask_count):
            if areas[first] > 0:
                overlaps[first, first] = 1.0
            for second in range(first + 1, mask_count):
                x_min: int = int(max(pixel_xyxy[first, 0], pixel_xyxy[second, 0]))
                y_min: int = int(max(pixel_xyxy[first, 1], pixel_xyxy[second, 1]))
                x_max: int = int(min(pixel_xyxy[first, 2], pixel_xyxy[second, 2]))
                y_max: int = int(min(pixel_xyxy[first, 3], pixel_xyxy[second, 3]))
                if x_max <= x_min or y_max <= y_min:
                    continue
                intersection: int = int(
                    np.logical_and(
                        masks[first, y_min:y_max, x_min:x_max], masks[second, y_min:y_max, x_min:x_max]
                    ).sum()
                )
                union: int = int(areas[first] + areas[second]) - intersection
                iou: float = intersection / union if union > 0 else 0.0
                overlaps[first, second] = iou
                overlaps[second, first] = iou
        return overlaps
