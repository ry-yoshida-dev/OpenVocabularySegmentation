from dataclasses import dataclass

from geometry import Box2D, Points2D
from image_container import BinaryImage
from open_vocabulary_detector import ImageSize

from ..array_types import BoolArray, FloatArray
from ..object_prompt import MaskLogits, ObjectPrompt
from .mask_contour import MaskContour


@dataclass(frozen=True, eq=False)
class ObjectSegmentation:
    """
    Mask predicted for one object prompt.

    Attributes
    ----------
    mask : BoolArray
        Pixels of the object in the source image, shape (H, W); empty when the model finds nothing.
    box : Box2D | None
        Bounding box of ``mask`` in absolute XYXY pixel coordinates with exclusive right and bottom edges; ``None``
        for an empty mask.
    score : float
        IoU of the mask with the object as predicted by the model, in ``[0, 1]``.
    stability_score : float
        IoU of the mask binarized slightly above and slightly below the mask threshold, in ``[0, 1]``; low for
        masks whose boundary is uncertain.
    prompt : ObjectPrompt
        Prompt the mask was predicted for.
    prompt_index : int
        Index of ``prompt`` among the prompts of the call.
    mask_logits : MaskLogits | None
        Low-resolution logits of the mask, the mask prompt of a refinement; ``None`` for backends without logits.
    """

    mask: BoolArray
    box: Box2D | None
    score: float
    stability_score: float
    prompt: ObjectPrompt
    prompt_index: int
    mask_logits: MaskLogits | None

    @property
    def area(self) -> int:
        """
        Number of pixels of the object.

        Returns
        -------
        int
            Count of ``True`` pixels in ``mask``.
        """
        return int(self.mask.sum())

    @property
    def is_empty(self) -> bool:
        """
        Whether the model found no pixel of the object.

        Returns
        -------
        bool
            True if ``mask`` has no ``True`` pixel.
        """
        return not bool(self.mask.any())

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
        Mask as an ``image_container.BinaryImage``, also usable as the mask prompt of another model.

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

    def refine(self, positive_points: Points2D | None = None, negative_points: Points2D | None = None) -> ObjectPrompt:
        """
        Build the prompt of the next refinement step: the points so far, corrective points and this mask's logits.

        Parameters
        ----------
        positive_points : Points2D | None, optional
            Points the mask missed.
        negative_points : Points2D | None, optional
            Points the mask wrongly covers.

        Returns
        -------
        ObjectPrompt
            Prompt with every point, the box of ``prompt`` and ``mask_logits`` as mask, or the mask of ``prompt``
            when there are no logits.
        """
        refined: ObjectPrompt = self.prompt.with_points(positive_points, negative_points)
        return refined if self.mask_logits is None else refined.with_mask(self.mask_logits)
