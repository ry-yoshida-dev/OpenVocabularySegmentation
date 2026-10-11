from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from geometry import Box2D, Box2DFormat, Points2D
from image_container import BinaryImage
from open_vocabulary_detector import ImageSize

from ..array_types import BoolArray, FloatArray
from ..options import PromptElement
from .mask_logits import MaskLogits
from .mask_prompt import MaskPrompt


@dataclass(frozen=True, eq=False)
class ObjectPrompt:
    """
    Points, box and mask telling a promptable model which single object to segment.

    Coordinates are absolute pixels of the image being segmented. Positive points lie on the object, negative points
    on what must be excluded; any combination of points, one box and one mask is accepted, as long as something
    marks the object (positive points, a box or a mask). A mask is either a binary mask of the image or the
    ``MaskLogits`` of a previous prediction, the way SAM refines a mask click by click.

    Attributes
    ----------
    positive_points : Points2D | None
        Points on the object, shape (P, 2) as ``(x, y)``.
    negative_points : Points2D | None
        Points off the object, shape (Q, 2) as ``(x, y)``.
    box : Box2D | None
        Box around the object in any ``Box2DFormat`` of ``geometry``.
    mask : MaskPrompt | None
        Binary mask of the image (``BinaryImage``) or logits of a previous prediction (``MaskLogits``).

    Raises
    ------
    ValueError
        If nothing marks the object, a point set is empty or not finite, or the box is not finite or has no area.
    """

    positive_points: Points2D | None = None
    negative_points: Points2D | None = None
    box: Box2D | None = None
    mask: MaskPrompt | None = None

    def __post_init__(self) -> None:
        if self.positive_points is None and self.box is None and self.mask is None:
            raise ValueError(
                "an object prompt needs positive points, a box or a mask; negative points only exclude regions."
            )
        for name, points in (("positive_points", self.positive_points), ("negative_points", self.negative_points)):
            if points is None:
                continue
            coordinates: FloatArray = self._coordinates_of(points)
            if not len(coordinates):
                raise ValueError(f"{name} must hold at least one point; pass None for no point.")
            if not bool(np.isfinite(coordinates).all()):
                raise ValueError(f"{name} must be finite. got {coordinates.tolist()}")
        xyxy: FloatArray | None = self.xyxy
        if xyxy is not None:
            if not bool(np.isfinite(xyxy).all()):
                raise ValueError(f"box must be finite. got {xyxy.tolist()}")
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                raise ValueError(f"box must have a positive width and height. got xyxy {xyxy.tolist()}")

    @classmethod
    def from_coordinates(
        cls,
        positive_points: Sequence[tuple[float, float]] = (),
        negative_points: Sequence[tuple[float, float]] = (),
        box_xyxy: tuple[float, float, float, float] | None = None,
        mask: MaskPrompt | None = None,
    ) -> "ObjectPrompt":
        """
        Build a prompt from plain coordinates, e.g. clicks of a user interface.

        Parameters
        ----------
        positive_points : Sequence[tuple[float, float]], optional
            ``(x, y)`` points on the object.
        negative_points : Sequence[tuple[float, float]], optional
            ``(x, y)`` points off the object.
        box_xyxy : tuple[float, float, float, float] | None, optional
            Box as ``(x_min, y_min, x_max, y_max)``.
        mask : MaskPrompt | None, optional
            Binary mask or logits of a previous prediction.

        Returns
        -------
        ObjectPrompt
            Prompt with the given elements.
        """
        return cls(
            positive_points=cls._points_of(positive_points),
            negative_points=cls._points_of(negative_points),
            box=(
                None
                if box_xyxy is None
                else Box2D.register(value=np.array(box_xyxy, dtype=np.float64), box2d_format=Box2DFormat.XYXY)
            ),
            mask=mask,
        )

    @property
    def elements(self) -> frozenset[PromptElement]:
        """
        Elements the prompt uses.

        Returns
        -------
        frozenset[PromptElement]
            ``POINT`` if any point is given, ``BOX`` and ``MASK`` if given.
        """
        elements: set[PromptElement] = set()
        if self.point_count:
            elements.add(PromptElement.POINT)
        if self.box is not None:
            elements.add(PromptElement.BOX)
        if self.mask is not None:
            elements.add(PromptElement.MASK)
        return frozenset(elements)

    @property
    def point_xy(self) -> FloatArray:
        """
        Every point, positive ones first.

        Returns
        -------
        FloatArray
            ``(x, y)`` pixel coordinates, shape (P + Q, 2).
        """
        return np.concatenate(
            [
                self._coordinates_of(points)
                for points in (self.positive_points, self.negative_points)
                if points is not None
            ]
            or [np.zeros((0, 2), dtype=np.float64)]
        )

    @property
    def is_point_positive(self) -> BoolArray:
        """
        Whether each point of ``point_xy`` is positive.

        Returns
        -------
        BoolArray
            Shape (P + Q,).
        """
        positive_count: int = 0 if self.positive_points is None else len(self.positive_points)
        negative_count: int = 0 if self.negative_points is None else len(self.negative_points)
        return np.concatenate([np.ones(positive_count, dtype=np.bool_), np.zeros(negative_count, dtype=np.bool_)])

    @property
    def point_count(self) -> int:
        """
        Number of points, positive and negative.

        Returns
        -------
        int
            P + Q.
        """
        return int(self.point_xy.shape[0])

    @property
    def xyxy(self) -> FloatArray | None:
        """
        Box as a plain array.

        Returns
        -------
        FloatArray | None
            Absolute XYXY pixel box, shape (4,); ``None`` without a box.
        """
        if self.box is None:
            return None
        return np.asarray(self.box.to_format(Box2DFormat.XYXY), dtype=np.float64).reshape(4)

    @property
    def is_ambiguous(self) -> bool:
        """
        Whether the prompt is a single point without box or mask, which may mean a whole, a part or a sub-part.

        Returns
        -------
        bool
            True for a single point only.
        """
        return self.point_count == 1 and self.box is None and self.mask is None

    def validate_for(self, image_size: ImageSize) -> None:
        """
        Check that the prompt lies on an image.

        Parameters
        ----------
        image_size : ImageSize
            Size of the image to segment.

        Raises
        ------
        ValueError
            If a point or the box lies outside the image, or the mask does not cover the image.
        """
        bounds: FloatArray = np.array([image_size.width, image_size.height], dtype=np.float64)
        point_xy: FloatArray = self.point_xy
        if bool(((point_xy < 0.0) | (point_xy > bounds)).any()):
            raise ValueError(
                f"points must lie inside the {image_size.width}x{image_size.height} image. got {point_xy.tolist()}"
            )
        xyxy: FloatArray | None = self.xyxy
        if xyxy is not None and bool(((xyxy < 0.0) | (xyxy > np.tile(bounds, 2))).any()):
            raise ValueError(
                f"box must lie inside the {image_size.width}x{image_size.height} image. got xyxy {xyxy.tolist()}"
            )
        match self.mask:
            case None | MaskLogits():
                pass
            case BinaryImage(value=value):
                if value.shape != (image_size.height, image_size.width):
                    raise ValueError(
                        f"mask must have shape ({image_size.height}, {image_size.width}). got {value.shape}"
                    )

    def with_points(
        self, positive_points: Points2D | None = None, negative_points: Points2D | None = None
    ) -> "ObjectPrompt":
        """
        Add points to the prompt, e.g. corrective clicks.

        Parameters
        ----------
        positive_points : Points2D | None, optional
            Further points on the object.
        negative_points : Points2D | None, optional
            Further points off the object.

        Returns
        -------
        ObjectPrompt
            Prompt with the points of both, the box and the mask of this prompt.
        """
        return replace(
            self,
            positive_points=self._merge(self.positive_points, positive_points),
            negative_points=self._merge(self.negative_points, negative_points),
        )

    def with_mask(self, mask: MaskPrompt | None) -> "ObjectPrompt":
        """
        Replace the mask of the prompt.

        Parameters
        ----------
        mask : MaskPrompt | None
            New mask, or ``None`` to drop it.

        Returns
        -------
        ObjectPrompt
            Prompt with the points and box of this prompt and ``mask``.
        """
        return replace(self, mask=mask)

    @staticmethod
    def _coordinates_of(points: Points2D) -> FloatArray:
        return np.asarray(points.value, dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def _points_of(coordinates: Sequence[tuple[float, float]]) -> Points2D | None:
        if not coordinates:
            return None
        return Points2D(value=np.array(coordinates, dtype=np.float64).reshape(-1, 2))

    @classmethod
    def _merge(cls, current: Points2D | None, added: Points2D | None) -> Points2D | None:
        if current is None:
            return added
        if added is None:
            return current
        return Points2D(value=np.concatenate([cls._coordinates_of(current), cls._coordinates_of(added)]))
