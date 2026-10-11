import math
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from open_vocabulary_detector import ImageSize
from PIL import Image, ImageDraw

from ...array_types import BoolArray, FloatArray


class PolygonRasterizer:
    """
    Fills polygons into a binary mask of the source image.

    Polygon points are continuous image coordinates, where pixel ``i`` spans ``[i, i + 1)``; a pixel is filled when
    its centre lies inside a polygon (even-odd rule), so a polygon covers as many pixels as its area. Polygons that
    contain no pixel centre, such as thin slivers, are drawn as their outline so they are not lost.
    """

    PIXEL_CENTER_OFFSET: ClassVar[float] = 0.5

    @classmethod
    def rasterize(cls, polygons: Sequence[FloatArray], image_size: ImageSize) -> BoolArray:
        """
        Fill the union of polygons.

        Parameters
        ----------
        polygons : Sequence[FloatArray]
            Polygon points as ``(x, y)`` relative to the image size, shape (P, 2) each.
        image_size : ImageSize
            Size of the mask.

        Returns
        -------
        BoolArray
            Pixels inside any polygon, shape (H, W); empty without polygons.
        """
        mask: BoolArray = np.zeros((image_size.height, image_size.width), dtype=np.bool_)
        scale: FloatArray = np.array([image_size.width, image_size.height], dtype=np.float64)
        for points in polygons:
            pixel_points: FloatArray = points * scale
            polygon_mask: BoolArray = cls._fill(pixel_points, image_size)
            if not polygon_mask.any():
                polygon_mask = cls._outline(pixel_points, image_size)
            mask |= polygon_mask
        return mask

    @classmethod
    def _fill(cls, points: FloatArray, image_size: ImageSize) -> BoolArray:
        mask: BoolArray = np.zeros((image_size.height, image_size.width), dtype=np.bool_)
        start_x: FloatArray = points[:, 0]
        start_y: FloatArray = points[:, 1]
        end_x: FloatArray = np.roll(start_x, -1)
        end_y: FloatArray = np.roll(start_y, -1)
        first_row: int = max(0, math.floor(float(start_y.min()) - cls.PIXEL_CENTER_OFFSET))
        last_row: int = min(image_size.height - 1, math.ceil(float(start_y.max())))
        for row in range(first_row, last_row + 1):
            center_y: float = row + cls.PIXEL_CENTER_OFFSET
            is_crossing: BoolArray = ((start_y <= center_y) & (center_y < end_y)) | (
                (end_y <= center_y) & (center_y < start_y)
            )
            if not is_crossing.any():
                continue
            crossing_ratio: FloatArray = (center_y - start_y[is_crossing]) / (end_y[is_crossing] - start_y[is_crossing])
            crossing_x: FloatArray = np.sort(
                start_x[is_crossing] + crossing_ratio * (end_x[is_crossing] - start_x[is_crossing])
            )
            for left_x, right_x in zip(crossing_x[0::2].tolist(), crossing_x[1::2].tolist(), strict=False):
                first_column: int = max(0, math.ceil(left_x - cls.PIXEL_CENTER_OFFSET))
                end_column: int = min(image_size.width, math.ceil(right_x - cls.PIXEL_CENTER_OFFSET))
                if first_column < end_column:
                    mask[row, first_column:end_column] = True
        return mask

    @classmethod
    def _outline(cls, points: FloatArray, image_size: ImageSize) -> BoolArray:
        canvas: Image.Image = Image.new("1", (image_size.width, image_size.height), 0)
        vertices: list[tuple[float, float]] = [
            (float(x), float(y)) for x, y in (points - cls.PIXEL_CENTER_OFFSET).tolist()
        ]
        ImageDraw.Draw(canvas).line(vertices + vertices[:1], fill=1)
        return np.array(canvas, dtype=np.bool_)
