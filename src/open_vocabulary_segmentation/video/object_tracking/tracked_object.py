from dataclasses import dataclass

from geometry import Box2D
from image_container import BinaryImage

from ...array_types import BoolArray, FloatArray
from ...result import MaskContour


@dataclass(frozen=True, eq=False)
class TrackedObject:
    """
    One tracked object on one video frame.

    Attributes
    ----------
    object_id : int
        Identifier the object was prompted with.
    frame_index : int
        Index of the frame in the video.
    mask : BoolArray
        Pixels of the object on the frame, shape (H, W); empty when the object is occluded or out of view.
    box : Box2D | None
        Bounding box of ``mask`` in absolute XYXY pixel coordinates with exclusive right and bottom edges; ``None``
        for an empty mask.
    presence_score : float
        Probability that the object is visible on the frame, in ``[0, 1]``.
    """

    object_id: int
    frame_index: int
    mask: BoolArray
    box: Box2D | None
    presence_score: float

    @property
    def area(self) -> int:
        """
        Number of pixels of the object on the frame.

        Returns
        -------
        int
            Count of ``True`` pixels in ``mask``.
        """
        return int(self.mask.sum())

    @property
    def is_visible(self) -> bool:
        """
        Whether the object has at least one pixel on the frame.

        Returns
        -------
        bool
            True if ``mask`` is not empty.
        """
        return bool(self.mask.any())

    @property
    def binary_image(self) -> BinaryImage:
        """
        Mask as an ``image_container.BinaryImage``, e.g. as the mask prompt of a correction.

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
