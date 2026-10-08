from dataclasses import dataclass

import numpy as np
from geometry import Box2D
from image_container import ArrayImageContainer, BinaryImage, ChannelOrder, ImageContainerType
from image_container.types import UInt8Image
from open_vocabulary_detector import ImageSize, PromptQuery

from ..array_types import BoolArray, FloatArray
from .mask_contour import MaskContour
from .source_image import SourceImage


@dataclass(frozen=True, eq=False)
class Segmentation:
    """
    Single segmented instance.

    Attributes
    ----------
    mask : BoolArray
        Pixels of the instance in the source image, shape (H, W).
    box : Box2D
        Bounding box of ``mask`` in absolute XYXY pixel coordinates; whole pixels with exclusive right and bottom
        edges.
    confidence : float
        Confidence in ``[0, 1]``.
    class_id : int
        Index into the classes of the originating prompt.
    class_name : str
        Output class name corresponding to ``class_id``.
    matched_query : PromptQuery
        Query that matched the instance, e.g. ``TextQuery("taxi")`` for class ``"car"``.
    """

    mask: BoolArray
    box: Box2D
    confidence: float
    class_id: int
    class_name: str
    matched_query: PromptQuery

    @property
    def area(self) -> int:
        """
        Number of pixels of the instance.

        Returns
        -------
        int
            Count of ``True`` pixels in ``mask``.
        """
        return int(self.mask.sum())

    @property
    def image_size(self) -> ImageSize:
        """
        Size of the source image the mask covers.

        Returns
        -------
        ImageSize
            Width and height of ``mask``.
        """
        height, width = self.mask.shape
        return ImageSize(width=width, height=height)

    @property
    def binary_image(self) -> BinaryImage:
        """
        Mask as an ``image_container.BinaryImage``, e.g. for connected components.

        Returns
        -------
        BinaryImage
            Copy of ``mask``.
        """
        return BinaryImage(value=self.mask.copy())

    @property
    def crop_slice(self) -> tuple[slice, slice]:
        """
        Rows and columns of the bounding box, in the ``(y_slice, x_slice)`` order of ``ImageContainer.crop``.

        Returns
        -------
        tuple[slice, slice]
            Slices selecting the bounding box of the instance.
        """
        bounds: list[float] = np.asarray(self.box.value, dtype=np.float64).reshape(4).tolist()
        x_min, y_min, x_max, y_max = (round(value) for value in bounds)
        return slice(y_min, y_max), slice(x_min, x_max)

    def crop(self, image: ImageContainerType) -> ImageContainerType:
        """
        Cut the bounding box of the instance out of the source image.

        Parameters
        ----------
        image : ImageContainerType
            Source image of the segmentation, any container type.

        Raises
        ------
        ValueError
            If the image size differs from the mask size.

        Returns
        -------
        ImageContainerType
            Crop of the same container type and channel order.
        """
        SourceImage.validate(image, self.image_size)
        return image.crop(self.crop_slice)

    def cutout(self, image: ImageContainerType, background_value: int = 0) -> ArrayImageContainer:
        """
        Cut the instance out of the source image, filling the pixels of its box outside the mask.

        Parameters
        ----------
        image : ImageContainerType
            Source image of the segmentation, any container type.
        background_value : int, optional
            Value in ``[0, 255]`` written to every channel of pixels outside the mask.

        Raises
        ------
        ValueError
            If the image size differs from the mask size or ``background_value`` is outside ``[0, 255]``.

        Returns
        -------
        ArrayImageContainer
            RGB crop of the bounding box showing only the instance.
        """
        if not 0 <= background_value <= 255:
            raise ValueError(f"background_value must be in [0, 255]. got {background_value}")
        pixels: UInt8Image = SourceImage.rgb_array(image, self.image_size)
        crop_slice: tuple[slice, slice] = self.crop_slice
        cutout: UInt8Image = pixels[crop_slice].copy()
        cutout[~self.mask[crop_slice]] = background_value
        return ArrayImageContainer(value=cutout, channel_order=ChannelOrder.RGB)

    def to_polygons(self) -> list[FloatArray]:
        """
        Outer boundary of every connected region of the mask, e.g. for COCO-style polygon export.

        Returns
        -------
        list[FloatArray]
            Polygon vertices as ``(x, y)`` pixel coordinates, shape (P, 2) each; holes are not traced.
        """
        return MaskContour.polygons(self.mask)
