import re
from typing import ClassVar

import numpy as np

from ...array_types import FloatArray


class PolygonParser:
    """
    Reads the polygons Florence-2 writes as location tokens into normalized points.

    For segmentation tasks Florence-2 writes the outline of one instance as ``(x, y)`` pairs of ``<loc_N>`` tokens.
    Several polygons of one instance are separated by ``<sep>`` or wrapped in ``<poly>...</poly>``. Each bin is
    read at its center, like the ``transformers`` post-processing, but without rounding to whole pixels. Parts with
    fewer than ``MIN_POINT_COUNT`` points enclose no area and are skipped.
    """

    BIN_COUNT: ClassVar[int] = 1000
    MIN_POINT_COUNT: ClassVar[int] = 3
    SEPARATOR_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"<sep>|</?poly>")
    LOCATION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"<loc_(\d+)>")

    @classmethod
    def parse(cls, generated_text: str) -> list[FloatArray]:
        """
        Extract every polygon of a generated answer.

        Parameters
        ----------
        generated_text : str
            Decoded generation, special tokens included.

        Returns
        -------
        list[FloatArray]
            Polygon points as ``(x, y)`` relative to the image size, shape (P, 2) each.
        """
        polygons: list[FloatArray] = []
        for part in cls.SEPARATOR_PATTERN.split(generated_text):
            bins: list[int] = [int(value) for value in cls.LOCATION_PATTERN.findall(part)]
            point_count: int = len(bins) // 2
            if point_count < cls.MIN_POINT_COUNT:
                continue
            points: FloatArray = np.array(bins[: 2 * point_count], dtype=np.float64).reshape(-1, 2)
            polygons.append(((points + 0.5) / cls.BIN_COUNT).clip(0.0, 1.0))
        return polygons
