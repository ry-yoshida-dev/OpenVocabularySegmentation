from collections.abc import Sequence

import numpy as np
from open_vocabulary_detector import ImageSize
from PIL import Image, ImageDraw

from ...array_types import BoolArray, FloatArray


class PolygonRasterizer:
    """
    Fills polygons into a binary mask of the source image.
    """

    @staticmethod
    def rasterize(polygons: Sequence[FloatArray], image_size: ImageSize) -> BoolArray:
        """
        Fill the union of polygons, outlines included.

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
        canvas: Image.Image = Image.new("1", (image_size.width, image_size.height), 0)
        drawing: ImageDraw.ImageDraw = ImageDraw.Draw(canvas)
        scale: FloatArray = np.array([image_size.width, image_size.height], dtype=np.float64)
        for points in polygons:
            pixel_points: list[tuple[float, float]] = [(float(x), float(y)) for x, y in (points * scale).tolist()]
            drawing.polygon(pixel_points, fill=1, outline=1)
        return np.array(canvas, dtype=np.bool_)
