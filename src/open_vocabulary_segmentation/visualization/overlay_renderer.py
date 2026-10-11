from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from image_container import ArrayImageContainer, ChannelOrder, ImageContainerType
from image_container.types import UInt8Image
from open_vocabulary_detector import ImageSize

from ..array_types import BoolArray, FloatArray, IntArray
from ..object_prompt import ObjectPrompt
from ..result import MaskContour, MaskGenerationResult, ObjectSegmentationResult, SegmentationResult, SourceImage
from ..video import ConceptFrame, TrackedFrame


@dataclass(frozen=True)
class MaskOverlayRenderer:
    """
    Draws masks over their source image: instances by class, prompted objects, tracked objects and tracks, generated
    regions.

    Every mask is tinted with a palette color and optionally outlined; masks drawn later cover earlier ones. Prompted
    objects can show their prompts: positive points, negative points and boxes.

    Attributes
    ----------
    opacity : float
        Weight of the mask color in tinted pixels, in ``[0, 1]``.
    contour_thickness : int
        Outline width in pixels; ``0`` draws no outline.
    palette : tuple[tuple[int, int, int], ...]
        RGB colors; color id ``k`` uses ``palette[k % len(palette)]``.
    positive_point_color : tuple[int, int, int]
        RGB color of positive prompt points.
    negative_point_color : tuple[int, int, int]
        RGB color of negative prompt points.
    point_radius : int
        Radius of prompt point markers in pixels.

    Raises
    ------
    ValueError
        If ``opacity`` is outside ``[0, 1]``, ``contour_thickness`` is negative, ``point_radius`` is not positive, or
        a color has a channel outside ``[0, 255]``.
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
    positive_point_color: tuple[int, int, int] = (0, 220, 0)
    negative_point_color: tuple[int, int, int] = (230, 0, 0)
    point_radius: int = 5

    def __post_init__(self) -> None:
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be in [0, 1]. got {self.opacity}")
        if self.contour_thickness < 0:
            raise ValueError(f"contour_thickness must not be negative. got {self.contour_thickness}")
        if self.point_radius <= 0:
            raise ValueError(f"point_radius must be positive. got {self.point_radius}")
        if not self.palette:
            raise ValueError("palette must have at least one color.")
        colors: list[tuple[int, int, int]] = [*self.palette, self.positive_point_color, self.negative_point_color]
        if any(not 0 <= channel <= 255 for color in colors for channel in color):
            raise ValueError(f"color channels must be in [0, 255]. got {colors}")

    def color_of(self, color_id: int) -> tuple[int, int, int]:
        """
        Palette color of a class, prompt, object or track.

        Parameters
        ----------
        color_id : int
            Non-negative identifier.

        Returns
        -------
        tuple[int, int, int]
            RGB color.
        """
        return self.palette[color_id % len(self.palette)]

    def render(self, image: ImageContainerType, result: SegmentationResult) -> ArrayImageContainer:
        """
        Draw every instance in its class color, the most confident instance on top.

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
        drawing_order: IntArray = np.argsort(result.confidences, kind="stable").astype(np.int64)
        return self.render_masks(image, result.image_size, result.masks, result.class_ids, drawing_order)

    def render_objects(
        self, image: ImageContainerType, result: ObjectSegmentationResult, is_prompt_drawn: bool = True
    ) -> ArrayImageContainer:
        """
        Draw the mask of every object prompt in the color of its prompt, optionally with the prompts.

        Parameters
        ----------
        image : ImageContainerType
            Source image of the result, any container type and channel order.
        result : ObjectSegmentationResult
            Masks to draw.
        is_prompt_drawn : bool, optional
            Whether to draw the boxes in the prompt color and the points in the point colors.

        Raises
        ------
        ValueError
            If the image size differs from the mask size.

        Returns
        -------
        ArrayImageContainer
            RGB image with tinted masks and prompts.
        """
        drawing_order: IntArray = np.argsort(result.scores, kind="stable").astype(np.int64)
        rendered: ArrayImageContainer = self.render_masks(
            image, result.image_size, result.masks, result.prompt_indices, drawing_order
        )
        if not is_prompt_drawn:
            return rendered
        pixels: UInt8Image = np.array(rendered.value, dtype=np.uint8, copy=True)
        for prompt_index, prompt in enumerate(result.prompts):
            self._draw_prompt(pixels, prompt, self.color_of(prompt_index))
        return ArrayImageContainer(value=pixels, channel_order=ChannelOrder.RGB)

    def render_tracked(self, image: ImageContainerType, frame: TrackedFrame) -> ArrayImageContainer:
        """
        Draw every tracked object of a video frame in the color of its object id.

        Parameters
        ----------
        image : ImageContainerType
            The video frame, any container type and channel order.
        frame : TrackedFrame
            Objects to draw.

        Raises
        ------
        ValueError
            If the image size differs from the mask size.

        Returns
        -------
        ArrayImageContainer
            RGB image with tinted objects.
        """
        drawing_order: IntArray = np.arange(len(frame), dtype=np.int64)
        return self.render_masks(image, frame.image_size, frame.masks, frame.object_ids, drawing_order)

    def render_concepts(self, image: ImageContainerType, frame: ConceptFrame) -> ArrayImageContainer:
        """
        Draw every tracked instance of a video frame in the color of its track, so identities stay recognizable.

        Parameters
        ----------
        image : ImageContainerType
            The video frame, any container type and channel order.
        frame : ConceptFrame
            Instances to draw.

        Raises
        ------
        ValueError
            If the image size differs from the mask size.

        Returns
        -------
        ArrayImageContainer
            RGB image with tinted instances.
        """
        instances: SegmentationResult = frame.instances
        drawing_order: IntArray = np.argsort(instances.confidences, kind="stable").astype(np.int64)
        return self.render_masks(image, instances.image_size, instances.masks, frame.track_ids, drawing_order)

    def render_generated(self, image: ImageContainerType, result: MaskGenerationResult) -> ArrayImageContainer:
        """
        Draw every generated region, large regions first so small ones stay visible.

        Parameters
        ----------
        image : ImageContainerType
            Source image of the result, any container type and channel order.
        result : MaskGenerationResult
            Regions to draw.

        Raises
        ------
        ValueError
            If the image size differs from the mask size.

        Returns
        -------
        ArrayImageContainer
            RGB image with tinted regions.
        """
        drawing_order: IntArray = np.argsort(-result.areas, kind="stable").astype(np.int64)
        return self.render_masks(
            image, result.image_size, result.masks, np.arange(len(result), dtype=np.int64), drawing_order
        )

    def render_masks(
        self,
        image: ImageContainerType,
        image_size: ImageSize,
        masks: BoolArray,
        color_ids: IntArray,
        drawing_order: IntArray,
    ) -> ArrayImageContainer:
        """
        Draw arbitrary masks in palette colors.

        Parameters
        ----------
        image : ImageContainerType
            Source image of the masks, any container type and channel order.
        image_size : ImageSize
            Size of the masks.
        masks : BoolArray
            Masks, shape (N, H, W).
        color_ids : IntArray
            Palette color id of each mask, shape (N,).
        drawing_order : IntArray
            Indices of the masks in drawing order; later masks cover earlier ones.

        Raises
        ------
        ValueError
            If the image size differs from ``image_size`` or the color ids do not match the masks.

        Returns
        -------
        ArrayImageContainer
            RGB image with tinted and outlined masks.
        """
        if color_ids.shape != (masks.shape[0],):
            raise ValueError(f"color_ids must have shape ({masks.shape[0]},). got {color_ids.shape}")
        canvas: FloatArray = SourceImage.rgb_array(image, image_size).astype(np.float64)
        order: list[int] = [int(index) for index in drawing_order]
        for index in order:
            mask: BoolArray = masks[index]
            color: FloatArray = np.array(self.color_of(int(color_ids[index])), dtype=np.float64)
            canvas[mask] = (1.0 - self.opacity) * canvas[mask] + self.opacity * color
        pixels: UInt8Image = np.ascontiguousarray(canvas.round().clip(0, 255).astype(np.uint8))
        if self.contour_thickness > 0:
            for index in order:
                contours: Sequence[np.ndarray] = [
                    polygon.round().astype(np.int32).reshape(-1, 1, 2) for polygon in MaskContour.polygons(masks[index])
                ]
                cv2.drawContours(pixels, contours, -1, self.color_of(int(color_ids[index])), self.contour_thickness)
        return ArrayImageContainer(value=pixels, channel_order=ChannelOrder.RGB)

    def _draw_prompt(self, pixels: UInt8Image, prompt: ObjectPrompt, color: tuple[int, int, int]) -> None:
        xyxy: FloatArray | None = prompt.xyxy
        if xyxy is not None:
            x_min, y_min, x_max, y_max = (round(float(value)) for value in xyxy)
            cv2.rectangle(pixels, (x_min, y_min), (x_max, y_max), color, max(1, self.contour_thickness))
        for (x, y), is_positive in zip(prompt.point_xy, prompt.is_point_positive, strict=True):
            center: tuple[int, int] = (round(float(x)), round(float(y)))
            point_color: tuple[int, int, int] = (
                self.positive_point_color if bool(is_positive) else self.negative_point_color
            )
            cv2.circle(pixels, center, self.point_radius, point_color, -1)
            cv2.circle(pixels, center, self.point_radius, (255, 255, 255), 1)
