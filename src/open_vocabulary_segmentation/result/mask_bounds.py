import numpy as np

from ..array_types import BoolArray, IntArray


class MaskBounds:
    """
    Pixel bounding boxes of binary masks.
    """

    @staticmethod
    def pixel_xyxy(masks: BoolArray) -> IntArray:
        """
        Bounding box of every mask as pixel bounds.

        Parameters
        ----------
        masks : BoolArray
            Binary masks, shape (N, H, W).

        Raises
        ------
        ValueError
            If the masks are not three-dimensional.

        Returns
        -------
        IntArray
            ``(x_min, y_min, x_max, y_max)`` with exclusive right and bottom edges, shape (N, 4); all zeros for an
            empty mask.
        """
        if masks.ndim != 3:
            raise ValueError(f"masks must have shape (N, H, W). got {masks.shape}")
        height, width = int(masks.shape[1]), int(masks.shape[2])
        is_column_filled: BoolArray = masks.any(axis=1)
        is_row_filled: BoolArray = masks.any(axis=2)
        bounds: IntArray = (
            np.stack(
                [
                    is_column_filled.argmax(axis=1),
                    is_row_filled.argmax(axis=1),
                    width - is_column_filled[:, ::-1].argmax(axis=1),
                    height - is_row_filled[:, ::-1].argmax(axis=1),
                ],
                axis=1,
            )
            .astype(np.int64)
            .reshape(-1, 4)
        )
        bounds[~is_column_filled.any(axis=1)] = 0
        return bounds
