import math
from dataclasses import dataclass

import numpy as np
from open_vocabulary_detector import ImageSize

from ..array_types import BoolArray, FloatArray, IntArray


@dataclass(frozen=True)
class CropBox:
    """
    One crop of the image segmented by automatic mask generation.

    Attributes
    ----------
    x_min : int
        Left edge in image pixels, inclusive.
    y_min : int
        Top edge in image pixels, inclusive.
    x_max : int
        Right edge in image pixels, exclusive.
    y_max : int
        Bottom edge in image pixels, exclusive.
    layer_index : int
        Crop layer; ``0`` is the whole image.

    Raises
    ------
    ValueError
        If the crop is empty or has a negative corner.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    layer_index: int

    def __post_init__(self) -> None:
        if self.x_min < 0 or self.y_min < 0 or self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(f"invalid crop ({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})")

    @classmethod
    def layout(cls, image_size: ImageSize, layer_count: int, overlap_ratio: float) -> list["CropBox"]:
        """
        Every crop of every layer, the whole image first, as in SAM's ``generate_crop_boxes``.

        Parameters
        ----------
        image_size : ImageSize
            Size of the image.
        layer_count : int
            Number of crop layers after the whole image.
        overlap_ratio : float
            Overlap of neighboring crops relative to the shorter image side.

        Returns
        -------
        list[CropBox]
            ``1 + 4 + 16 + ...`` crops.
        """
        crops: list[CropBox] = [cls(x_min=0, y_min=0, x_max=image_size.width, y_max=image_size.height, layer_index=0)]
        short_side: int = min(image_size.width, image_size.height)
        for layer_index in range(1, layer_count + 1):
            crops_per_side: int = 2**layer_index
            overlap: int = int(overlap_ratio * short_side * (2 / crops_per_side))
            crop_width: int = math.ceil((overlap * (crops_per_side - 1) + image_size.width) / crops_per_side)
            crop_height: int = math.ceil((overlap * (crops_per_side - 1) + image_size.height) / crops_per_side)
            for column in range(crops_per_side):
                for row in range(crops_per_side):
                    x_min: int = (crop_width - overlap) * column
                    y_min: int = (crop_height - overlap) * row
                    crops.append(
                        cls(
                            x_min=x_min,
                            y_min=y_min,
                            x_max=min(image_size.width, x_min + crop_width),
                            y_max=min(image_size.height, y_min + crop_height),
                            layer_index=layer_index,
                        )
                    )
        return crops

    @property
    def size(self) -> ImageSize:
        """
        Size of the crop.

        Returns
        -------
        ImageSize
            Width and height in pixels.
        """
        return ImageSize(width=self.x_max - self.x_min, height=self.y_max - self.y_min)

    @property
    def xyxy(self) -> IntArray:
        """
        Crop as pixel bounds.

        Returns
        -------
        IntArray
            ``(x_min, y_min, x_max, y_max)``, shape (4,).
        """
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max], dtype=np.int64)

    @property
    def area(self) -> int:
        """
        Number of pixels of the crop.

        Returns
        -------
        int
            Width times height.
        """
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def point_grid(self, points_per_side: int) -> FloatArray:
        """
        Regular grid of points centered in the cells of the crop, in crop pixels.

        Parameters
        ----------
        points_per_side : int
            Points along each side.

        Returns
        -------
        FloatArray
            ``(x, y)`` points, shape (points_per_side ** 2, 2).
        """
        offsets: FloatArray = (np.arange(points_per_side, dtype=np.float64) + 0.5) / points_per_side
        grid_x, grid_y = np.meshgrid(offsets, offsets)
        normalized: FloatArray = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
        return normalized * np.array([self.size.width, self.size.height], dtype=np.float64)

    def is_cutting(self, pixel_xyxy: IntArray, image_size: ImageSize, tolerance: float = 20.0) -> BoolArray:
        """
        Whether boxes lie on an edge of the crop that is not near an edge of the image, i.e. their objects are cut by
        the crop, as in SAM's ``is_box_near_crop_edge``.

        Parameters
        ----------
        pixel_xyxy : IntArray
            Boxes in image pixels, shape (N, 4).
        image_size : ImageSize
            Size of the whole image.
        tolerance : float, optional
            Distance in pixels within which a box side counts as touching an edge.

        Returns
        -------
        BoolArray
            Shape (N,).
        """
        boxes: FloatArray = pixel_xyxy.astype(np.float64).reshape(-1, 4)
        image_xyxy: FloatArray = np.array([0, 0, image_size.width, image_size.height], dtype=np.float64)
        is_near_crop_edge: BoolArray = np.abs(boxes - self.xyxy.astype(np.float64)) <= tolerance
        is_near_image_edge: BoolArray = np.abs(boxes - image_xyxy) <= tolerance
        return (is_near_crop_edge & ~is_near_image_edge).any(axis=1)
