from dataclasses import dataclass

import cv2
import numpy as np
from image_container import ArrayImageContainer, ChannelOrder, ImageContainerType

from ..array_types import FloatArray
from ..result import MaskContour, SegmentationResult, SourceImage


@dataclass(frozen=True)
class MaskOverlayRenderer:
    """
    Draws the instances of a segmentation result over its source image.

    Every instance is tinted with the color of its class and optionally outlined. Less confident instances are drawn
    first, so the most confident one stays visible where instances overlap.

    Attributes
    ----------
    opacity : float
        Weight of the class color in tinted pixels, in ``[0, 1]``.
    contour_thickness : int
        Outline width in pixels; ``0`` draws no outline.
    palette : tuple[tuple[int, int, int], ...]
        RGB colors; class ``k`` uses ``palette[k % len(palette)]``.

    Raises
    ------
    ValueError
        If ``opacity`` is outside ``[0, 1]``, ``contour_thickness`` is negative, or the palette is empty or has a
        channel outside ``[0, 255]``.
    """

    opacity: float = 0.5
    contour_thickness: int = 2
    palette: tuple[tuple[int, int, int], ...] = (
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
        (0, 128, 128),
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be in [0, 1]. got {self.opacity}")
        if self.contour_thickness < 0:
            raise ValueError(f"contour_thickness must not be negative. got {self.contour_thickness}")
        if not self.palette:
            raise ValueError("palette must have at least one color.")
        if any(not 0 <= channel <= 255 for color in self.palette for channel in color):
            raise ValueError(f"palette channels must be in [0, 255]. got {self.palette}")

    def color_of(self, class_id: int) -> tuple[int, int, int]:
        """
        Color of a class.

        Parameters
        ----------
        class_id : int
            Class id of the prompt.

        Returns
        -------
        tuple[int, int, int]
            RGB color.
        """
        return self.palette[class_id % len(self.palette)]

    def render(self, image: ImageContainerType, result: SegmentationResult) -> ArrayImageContainer:
        """
        Draw every instance of a result over its source image.

        Parameters
        ----------
        image : ImageContainerType
            Source image of the result, any container type and channel order.
        result : SegmentationResult
            Instances to draw.

        Raises
        ------
        ValueError
            If the image size differs from the mask size.

        Returns
        -------
        ArrayImageContainer
            RGB image with tinted and outlined instances.
        """
        canvas: FloatArray = SourceImage.rgb_array(image, result.image_size).astype(np.float64)
        class_ids: list[int] = result.class_ids.tolist()
        drawing_order: list[int] = np.argsort(result.confidences, kind="stable").tolist()
        for index in drawing_order:
            mask = result.masks[index]
            color: FloatArray = np.array(self.color_of(class_ids[index]), dtype=np.float64)
            canvas[mask] = (1.0 - self.opacity) * canvas[mask] + self.opacity * color
        pixels = np.ascontiguousarray(canvas.round().clip(0, 255).astype(np.uint8))
        if self.contour_thickness > 0:
            for index in drawing_order:
                contours: list[np.ndarray] = [
                    polygon.round().astype(np.int32).reshape(-1, 1, 2)
                    for polygon in MaskContour.polygons(result.masks[index])
                ]
                cv2.drawContours(pixels, contours, -1, self.color_of(class_ids[index]), self.contour_thickness)
        return ArrayImageContainer(value=pixels, channel_order=ChannelOrder.RGB)
