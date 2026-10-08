from typing import ClassVar

import numpy as np

from ...array_types import FloatArray, IntArray


class LocationTokens:
    """
    Writes a box as the location tokens Florence-2 reads in region tasks (``<loc_8><loc_114><loc_497><loc_990>``).

    Each coordinate relative to the image size is quantized into ``BIN_COUNT`` bins, the inverse of reading a bin
    at its center, so a box parsed from a generation is written back to the same tokens.
    """

    BIN_COUNT: ClassVar[int] = 1000

    @classmethod
    def encode_box(cls, normalized_xyxy: FloatArray) -> str:
        """
        Write a box as four location tokens.

        Parameters
        ----------
        normalized_xyxy : FloatArray
            XYXY box relative to the image size, shape (4,).

        Raises
        ------
        ValueError
            If the box does not have four coordinates.

        Returns
        -------
        str
            Four consecutive ``<loc_N>`` tokens.
        """
        if normalized_xyxy.shape != (4,):
            raise ValueError(f"normalized_xyxy must have shape (4,). got {normalized_xyxy.shape}")
        bins: IntArray = np.clip(np.floor(normalized_xyxy * cls.BIN_COUNT), 0, cls.BIN_COUNT - 1).astype(np.int64)
        return "".join(f"<loc_{bin_index}>" for bin_index in bins.tolist())
