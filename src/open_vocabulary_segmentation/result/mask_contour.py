from typing import ClassVar

import cv2
import numpy as np

from ..array_types import BoolArray, FloatArray


class MaskContour:
    """
    Traces the outer boundaries of a binary mask as polygons.
    """

    MIN_POINT_COUNT: ClassVar[int] = 3

    @classmethod
    def polygons(cls, mask: BoolArray) -> list[FloatArray]:
        """
        Outer boundary of every connected region of a mask.

        Holes are not traced; regions whose boundary has fewer than ``MIN_POINT_COUNT`` points (single pixels and
        one-pixel lines) are skipped.

        Parameters
        ----------
        mask : BoolArray
            Binary mask, shape (H, W).

        Raises
        ------
        ValueError
            If the mask is not two-dimensional.

        Returns
        -------
        list[FloatArray]
            Polygon vertices as ``(x, y)`` pixel coordinates, shape (P, 2) each.
        """
        if mask.ndim != 2:
            raise ValueError(f"mask must have shape (H, W). got {mask.shape}")
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            np.asarray(contour, dtype=np.float64).reshape(-1, 2)
            for contour in contours
            if len(contour) >= cls.MIN_POINT_COUNT
        ]
