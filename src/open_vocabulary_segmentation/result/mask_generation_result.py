from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from geometry import Box2D, Box2DFormat, Boxes2D
from image_container import BinaryImage
from open_vocabulary_detector import ImageSize

from ..array_types import BoolArray, FloatArray, IntArray
from .generated_mask import GeneratedMask
from .mask_bounds import MaskBounds


@dataclass(frozen=True, eq=False)
class MaskGenerationResult:
    """
    Every region automatic mask generation found in one image, without classes.

    Attributes
    ----------
    masks : BoolArray
        Pixels of each region, shape (N, H, W).
    scores : FloatArray
        Predicted IoU of each mask in ``[0, 1]``, shape (N,).
    stability_scores : FloatArray
        Stability score of each mask in ``[0, 1]``, shape (N,).
    point_xy : FloatArray
        Grid point that prompted each mask in image pixels, shape (N, 2).
    crop_xyxy : FloatArray
        Crop each mask was predicted in as image pixel bounds, shape (N, 4).
    image_size : ImageSize
        Size of the source image.
    pixel_xyxy : IntArray
        Bounding box of each mask as pixel bounds with exclusive right and bottom edges, shape (N, 4).

    Raises
    ------
    ValueError
        If the masks do not match the image size, a mask is empty, or array lengths disagree.
    """

    masks: BoolArray
    scores: FloatArray
    stability_scores: FloatArray
    point_xy: FloatArray
    crop_xyxy: FloatArray
    image_size: ImageSize
    pixel_xyxy: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        expected_shape: tuple[int, int] = (self.image_size.height, self.image_size.width)
        if self.masks.ndim != 3 or self.masks.shape[1:] != expected_shape:
            raise ValueError(
                f"masks must have shape (N, {expected_shape[0]}, {expected_shape[1]}). got {self.masks.shape}"
            )
        mask_count: int = int(self.masks.shape[0])
        for name, values, expected in (
            ("scores", self.scores, (mask_count,)),
            ("stability_scores", self.stability_scores, (mask_count,)),
            ("point_xy", self.point_xy, (mask_count, 2)),
            ("crop_xyxy", self.crop_xyxy, (mask_count, 4)),
        ):
            if values.shape != expected:
                raise ValueError(f"{name} must have shape {expected}. got {values.shape}")
        if not bool(self.masks.any(axis=(1, 2)).all()):
            raise ValueError("generated masks must not be empty.")
        object.__setattr__(self, "pixel_xyxy", MaskBounds.pixel_xyxy(self.masks))

    @property
    def boxes(self) -> Boxes2D:
        """
        Bounding box of every mask.

        Returns
        -------
        Boxes2D
            Absolute XYXY pixel boxes, shape (N, 4).
        """
        return Boxes2D.register(value=self.pixel_xyxy.astype(np.float64), box2d_format=Box2DFormat.XYXY)

    @property
    def areas(self) -> IntArray:
        """
        Pixel count of every mask.

        Returns
        -------
        IntArray
            Shape (N,).
        """
        return self.masks.sum(axis=(1, 2)).astype(np.int64)

    def __len__(self) -> int:
        return int(self.masks.shape[0])

    def __getitem__(self, index: int) -> GeneratedMask:
        if not -len(self) <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} masks.")
        position: int = index % len(self)
        return GeneratedMask(
            mask=self.masks[position],
            box=Box2D.register(value=self.pixel_xyxy[position].astype(np.float64), box2d_format=Box2DFormat.XYXY),
            score=float(self.scores[position]),
            stability_score=float(self.stability_scores[position]),
            point_xy=self.point_xy[position],
            crop_xyxy=self.crop_xyxy[position],
        )

    def __iter__(self) -> Iterator[GeneratedMask]:
        for index in range(len(self)):
            yield self[index]

    def select(self, indices: BoolArray | IntArray) -> "MaskGenerationResult":
        """
        Keep a subset of masks.

        Parameters
        ----------
        indices : BoolArray | IntArray
            Boolean mask of shape (N,) or integer indices.

        Returns
        -------
        MaskGenerationResult
            Masks at the given indices, in the given order.
        """
        return MaskGenerationResult(
            masks=self.masks[indices].reshape(-1, self.image_size.height, self.image_size.width),
            scores=self.scores[indices].reshape(-1),
            stability_scores=self.stability_scores[indices].reshape(-1),
            point_xy=self.point_xy[indices].reshape(-1, 2),
            crop_xyxy=self.crop_xyxy[indices].reshape(-1, 4),
            image_size=self.image_size,
        )

    def sort_by_area(self, is_descending: bool = True) -> "MaskGenerationResult":
        """
        Order masks by pixel count, e.g. to draw large regions first.

        Parameters
        ----------
        is_descending : bool, optional
            Whether the largest mask comes first.

        Returns
        -------
        MaskGenerationResult
            Sorted masks.
        """
        areas: IntArray = self.areas
        return self.select(np.argsort(-areas if is_descending else areas, kind="stable").astype(np.int64))

    def sort_by_score(self) -> "MaskGenerationResult":
        """
        Order masks by descending predicted IoU.

        Returns
        -------
        MaskGenerationResult
            Sorted masks.
        """
        return self.select(np.argsort(-self.scores, kind="stable").astype(np.int64))

    def filter_by_area(self, minimum_area: int, maximum_area: int | None = None) -> "MaskGenerationResult":
        """
        Keep masks whose pixel count lies in a range.

        Parameters
        ----------
        minimum_area : int
            Minimum number of pixels, inclusive.
        maximum_area : int | None, optional
            Maximum number of pixels, inclusive; ``None`` sets no upper bound.

        Raises
        ------
        ValueError
            If ``minimum_area`` is negative or exceeds ``maximum_area``.

        Returns
        -------
        MaskGenerationResult
            Masks within the range, in the current order.
        """
        if minimum_area < 0:
            raise ValueError(f"minimum_area must not be negative. got {minimum_area}")
        if maximum_area is not None and maximum_area < minimum_area:
            raise ValueError(f"maximum_area must be at least minimum_area. got {maximum_area} < {minimum_area}")
        areas: IntArray = self.areas
        is_kept: BoolArray = areas >= minimum_area
        if maximum_area is not None:
            is_kept &= areas <= maximum_area
        return self.select(is_kept)

    def union_mask(self) -> BinaryImage:
        """
        Union of every mask.

        Returns
        -------
        BinaryImage
            Pixels covered by any region.
        """
        return BinaryImage(value=self.masks.any(axis=0))

    def instance_map(self) -> IntArray:
        """
        Label map of the regions; where regions overlap, the smaller one claims the pixel.

        Returns
        -------
        IntArray
            Index of the mask in this result for each pixel, ``-1`` where no region lies, shape (H, W).
        """
        instance_map: IntArray = np.full((self.image_size.height, self.image_size.width), -1, dtype=np.int64)
        for index in np.argsort(-self.areas, kind="stable").tolist():
            instance_map[self.masks[index]] = index
        return instance_map

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(masks={len(self)}, "
            + f"image_size=({self.image_size.width}, {self.image_size.height}))"
        )
