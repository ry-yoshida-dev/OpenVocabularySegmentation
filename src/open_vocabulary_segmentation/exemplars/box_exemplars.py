from dataclasses import dataclass

import numpy as np
from geometry import Box2DFormat, Boxes2D
from open_vocabulary_detector import ImageSize

from ..array_types import BoolArray, FloatArray


@dataclass(frozen=True, eq=False)
class BoxExemplars:
    """
    Boxes in the image being segmented showing what a class is (positive) and is not (negative).

    SAM 3 reads exemplars together with the text of each query of the class and segments every instance of the
    concept they define, not only the boxed ones. With ``is_text_ignored`` the class is segmented from the boxes
    alone, e.g. "everything that looks like this", and reported under its first query.

    Attributes
    ----------
    positive_boxes : Boxes2D | None
        Boxes around instances of the class, any ``Box2DFormat``.
    negative_boxes : Boxes2D | None
        Boxes around look-alikes that are not the class, any ``Box2DFormat``.
    is_text_ignored : bool
        Whether to segment from the boxes alone instead of the text queries plus the boxes.

    Raises
    ------
    ValueError
        If there is no box, a box set is empty, a box is not finite or has no area, or the text is ignored without
        a positive box.
    """

    positive_boxes: Boxes2D | None = None
    negative_boxes: Boxes2D | None = None
    is_text_ignored: bool = False

    def __post_init__(self) -> None:
        if self.positive_boxes is None and self.negative_boxes is None:
            raise ValueError("box exemplars need positive or negative boxes.")
        if self.is_text_ignored and self.positive_boxes is None:
            raise ValueError("ignoring the text needs at least one positive box to define the class.")
        for name, boxes in (("positive_boxes", self.positive_boxes), ("negative_boxes", self.negative_boxes)):
            if boxes is None:
                continue
            xyxy: FloatArray = self._xyxy_of(boxes)
            if not len(xyxy):
                raise ValueError(f"{name} must hold at least one box; pass None for no box.")
            if not bool(np.isfinite(xyxy).all()):
                raise ValueError(f"{name} must be finite. got {xyxy.tolist()}")
            if bool(((xyxy[:, 2] <= xyxy[:, 0]) | (xyxy[:, 3] <= xyxy[:, 1])).any()):
                raise ValueError(f"{name} must have a positive width and height. got xyxy {xyxy.tolist()}")

    @property
    def xyxy(self) -> FloatArray:
        """
        Every box, positive ones first.

        Returns
        -------
        FloatArray
            Absolute XYXY pixel boxes, shape (N, 4).
        """
        return np.concatenate(
            [self._xyxy_of(boxes) for boxes in (self.positive_boxes, self.negative_boxes) if boxes is not None]
        )

    @property
    def is_positive(self) -> BoolArray:
        """
        Whether each box of ``xyxy`` is positive.

        Returns
        -------
        BoolArray
            Shape (N,).
        """
        positive_count: int = 0 if self.positive_boxes is None else len(self.positive_boxes)
        negative_count: int = 0 if self.negative_boxes is None else len(self.negative_boxes)
        return np.concatenate([np.ones(positive_count, dtype=np.bool_), np.zeros(negative_count, dtype=np.bool_)])

    def validate_for(self, image_size: ImageSize) -> None:
        """
        Check that every box lies on an image.

        Parameters
        ----------
        image_size : ImageSize
            Size of the image to segment.

        Raises
        ------
        ValueError
            If a box lies outside the image.
        """
        bounds: FloatArray = np.array(
            [image_size.width, image_size.height, image_size.width, image_size.height], dtype=np.float64
        )
        xyxy: FloatArray = self.xyxy
        if bool(((xyxy < 0.0) | (xyxy > bounds)).any()):
            raise ValueError(
                f"exemplar boxes must lie inside the {image_size.width}x{image_size.height} image. "
                + f"got xyxy {xyxy.tolist()}"
            )

    @staticmethod
    def _xyxy_of(boxes: Boxes2D) -> FloatArray:
        converted: FloatArray = np.asarray(boxes.to_format(Box2DFormat.XYXY), dtype=np.float64)
        return converted.reshape(-1, 4)
