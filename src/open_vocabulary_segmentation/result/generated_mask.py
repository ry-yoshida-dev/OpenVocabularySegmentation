from dataclasses import dataclass

from geometry import Box2D
from image_container import BinaryImage

from ..array_types import BoolArray, FloatArray
from .mask_contour import MaskContour


@dataclass(frozen=True, eq=False)
class GeneratedMask:
    """
    One mask found by automatic mask generation.

    Attributes
    ----------
    mask : BoolArray
        Pixels of the region in the source image, shape (H, W).
    box : Box2D
        Bounding box of ``mask`` in absolute XYXY pixel coordinates with exclusive right and bottom edges.
    score : float
        Predicted IoU of the mask in ``[0, 1]``.
    stability_score : float
        Stability score of the mask in ``[0, 1]``.
    point_xy : FloatArray
        Grid point that prompted the mask, in image pixels, shape (2,).
    crop_xyxy : FloatArray
        Crop the mask was predicted in, as image pixel bounds, shape (4,).
    """

    mask: BoolArray
    box: Box2D
    score: float
    stability_score: float
    point_xy: FloatArray
    crop_xyxy: FloatArray

    @property
    def area(self) -> int:
        """
        Number of pixels of the region.

        Returns
        -------
        int
            Count of ``True`` pixels in ``mask``.
        """
        return int(self.mask.sum())

    @property
    def binary_image(self) -> BinaryImage:
        """
        Mask as an ``image_container.BinaryImage``, also usable as the mask prompt of a promptable segmenter.

        Returns
        -------
        BinaryImage
            Copy of ``mask``.
        """
        return BinaryImage(value=self.mask.copy())

    def to_polygons(self) -> list[FloatArray]:
        """
        Outer boundary of every connected region of the mask.

        Returns
        -------
        list[FloatArray]
            Polygon vertices as ``(x, y)`` pixel coordinates, shape (P, 2) each; holes are not traced.
        """
        return MaskContour.polygons(self.mask)
