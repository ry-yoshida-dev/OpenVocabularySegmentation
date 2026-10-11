from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
from geometry import Box2D, Box2DFormat
from image_container import BinaryImage
from open_vocabulary_detector import ImageSize

from ..array_types import BoolArray, FloatArray, IntArray
from ..object_prompt import MaskLogits, ObjectPrompt
from .mask_bounds import MaskBounds
from .mask_suppression import MaskSuppression
from .object_segmentation import ObjectSegmentation


@dataclass(frozen=True, eq=False)
class ObjectSegmentationResult:
    """
    Masks predicted for the object prompts of one image.

    Every prompt yields one mask, or three with ``MaskOutput.ALL_CANDIDATES``; ``prompt_indices`` maps each mask to
    its prompt. Masks may be empty when the model finds nothing, unlike the instances of a ``SegmentationResult``.

    Attributes
    ----------
    masks : BoolArray
        Pixels of each mask in the source image, shape (N, H, W).
    scores : FloatArray
        Predicted IoU of each mask in ``[0, 1]``, shape (N,).
    stability_scores : FloatArray
        Stability score of each mask in ``[0, 1]``, shape (N,).
    prompt_indices : IntArray
        Prompt of each mask, indexing ``prompts``, shape (N,).
    prompts : tuple[ObjectPrompt, ...]
        Prompts of the call, in input order.
    mask_logits : tuple[MaskLogits | None, ...]
        Low-resolution logits of each mask, length N; ``None`` for backends without logits (Florence-2).
    image_size : ImageSize
        Size of the image the masks cover.
    pixel_xyxy : IntArray
        Bounding box of each mask as pixel bounds with exclusive right and bottom edges, all zeros for an empty
        mask, shape (N, 4).

    Raises
    ------
    ValueError
        If the masks do not match the image size, array lengths disagree, a score is outside ``[0, 1]`` or a
        prompt index is out of range.
    """

    masks: BoolArray
    scores: FloatArray
    stability_scores: FloatArray
    prompt_indices: IntArray
    prompts: tuple[ObjectPrompt, ...]
    mask_logits: tuple[MaskLogits | None, ...]
    image_size: ImageSize
    pixel_xyxy: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        expected_shape: tuple[int, int] = (self.image_size.height, self.image_size.width)
        if self.masks.ndim != 3 or self.masks.shape[1:] != expected_shape:
            raise ValueError(
                f"masks must have shape (N, {expected_shape[0]}, {expected_shape[1]}). got {self.masks.shape}"
            )
        if self.masks.dtype != np.bool_:
            raise ValueError(f"masks must be boolean. got {self.masks.dtype}")
        mask_count: int = int(self.masks.shape[0])
        for name, values in (
            ("scores", self.scores),
            ("stability_scores", self.stability_scores),
            ("prompt_indices", self.prompt_indices),
        ):
            if values.shape != (mask_count,):
                raise ValueError(f"{name} must have shape ({mask_count},). got {values.shape}")
        if len(self.mask_logits) != mask_count:
            raise ValueError(f"mask_logits must have {mask_count} entries. got {len(self.mask_logits)}")
        for name, values in (("scores", self.scores), ("stability_scores", self.stability_scores)):
            if bool(((values < 0.0) | (values > 1.0)).any()):
                raise ValueError(f"{name} must be in [0, 1]. got {values.tolist()}")
        if mask_count and (self.prompt_indices.min() < 0 or self.prompt_indices.max() >= len(self.prompts)):
            raise ValueError(f"prompt_indices must be in [0, {len(self.prompts)}). got {self.prompt_indices}")
        object.__setattr__(self, "pixel_xyxy", MaskBounds.pixel_xyxy(self.masks))

    @classmethod
    def empty(cls, prompts: Sequence[ObjectPrompt], image_size: ImageSize) -> "ObjectSegmentationResult":
        """
        Build a result without masks.

        Parameters
        ----------
        prompts : Sequence[ObjectPrompt]
            Prompts of the call.
        image_size : ImageSize
            Size of the source image.

        Returns
        -------
        ObjectSegmentationResult
            Result with zero masks.
        """
        return cls(
            masks=np.zeros((0, image_size.height, image_size.width), dtype=np.bool_),
            scores=np.zeros((0,), dtype=np.float64),
            stability_scores=np.zeros((0,), dtype=np.float64),
            prompt_indices=np.zeros((0,), dtype=np.int64),
            prompts=tuple(prompts),
            mask_logits=(),
            image_size=image_size,
        )

    @property
    def areas(self) -> IntArray:
        """
        Pixel count of every mask.

        Returns
        -------
        IntArray
            Number of ``True`` pixels of each mask, shape (N,).
        """
        return self.masks.sum(axis=(1, 2)).astype(np.int64)

    @property
    def is_empty_mask(self) -> BoolArray:
        """
        Whether each mask has no pixel.

        Returns
        -------
        BoolArray
            Shape (N,).
        """
        return ~self.masks.any(axis=(1, 2))

    def __len__(self) -> int:
        return int(self.masks.shape[0])

    def __getitem__(self, index: int) -> ObjectSegmentation:
        if not -len(self) <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} masks.")
        position: int = index % len(self)
        prompt_index: int = int(self.prompt_indices[position])
        is_empty: bool = bool(self.is_empty_mask[position])
        return ObjectSegmentation(
            mask=self.masks[position],
            box=(
                None
                if is_empty
                else Box2D.register(value=self.pixel_xyxy[position].astype(np.float64), box2d_format=Box2DFormat.XYXY)
            ),
            score=float(self.scores[position]),
            stability_score=float(self.stability_scores[position]),
            prompt=self.prompts[prompt_index],
            prompt_index=prompt_index,
            mask_logits=self.mask_logits[position],
        )

    def __iter__(self) -> Iterator[ObjectSegmentation]:
        for index in range(len(self)):
            yield self[index]

    def select(self, indices: BoolArray | IntArray) -> "ObjectSegmentationResult":
        """
        Keep a subset of masks.

        Parameters
        ----------
        indices : BoolArray | IntArray
            Boolean mask of shape (N,) or integer indices.

        Returns
        -------
        ObjectSegmentationResult
            Masks at the given indices, in the given order, with every prompt of the call.
        """
        positions: IntArray = np.arange(len(self), dtype=np.int64)[indices].reshape(-1)
        return ObjectSegmentationResult(
            masks=self.masks[positions].reshape(-1, self.image_size.height, self.image_size.width),
            scores=self.scores[positions],
            stability_scores=self.stability_scores[positions],
            prompt_indices=self.prompt_indices[positions],
            prompts=self.prompts,
            mask_logits=tuple(self.mask_logits[int(positions[index])] for index in range(len(positions))),
            image_size=self.image_size,
        )

    def of_prompt(self, prompt_index: int) -> "ObjectSegmentationResult":
        """
        Keep the masks of one prompt.

        Parameters
        ----------
        prompt_index : int
            Index of the prompt among ``prompts``.

        Raises
        ------
        IndexError
            If ``prompt_index`` is out of range.

        Returns
        -------
        ObjectSegmentationResult
            Masks of the prompt, in the current order.
        """
        if not 0 <= prompt_index < len(self.prompts):
            raise IndexError(f"prompt_index {prompt_index} is out of range for {len(self.prompts)} prompts.")
        return self.select(self.prompt_indices == prompt_index)

    def filter_by_score(self, minimum_score: float) -> "ObjectSegmentationResult":
        """
        Keep masks whose predicted IoU is at least ``minimum_score``.

        Parameters
        ----------
        minimum_score : float
            Minimum predicted IoU.

        Returns
        -------
        ObjectSegmentationResult
            Filtered masks.
        """
        return self.select(self.scores >= minimum_score)

    def filter_by_stability(self, minimum_stability_score: float) -> "ObjectSegmentationResult":
        """
        Keep masks whose stability score is at least ``minimum_stability_score``.

        Parameters
        ----------
        minimum_stability_score : float
            Minimum stability score.

        Returns
        -------
        ObjectSegmentationResult
            Filtered masks.
        """
        return self.select(self.stability_scores >= minimum_stability_score)

    def filter_by_area(self, minimum_area: int, maximum_area: int | None = None) -> "ObjectSegmentationResult":
        """
        Keep masks whose pixel count lies in a range, e.g. to drop empty masks with ``minimum_area=1``.

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
        ObjectSegmentationResult
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

    def sort_by_score(self) -> "ObjectSegmentationResult":
        """
        Order masks by descending predicted IoU.

        Returns
        -------
        ObjectSegmentationResult
            Sorted masks.
        """
        return self.select(np.argsort(-self.scores, kind="stable").astype(np.int64))

    def best_per_prompt(self) -> "ObjectSegmentationResult":
        """
        Keep the mask with the highest predicted IoU of every prompt, e.g. after ``MaskOutput.ALL_CANDIDATES``.

        Returns
        -------
        ObjectSegmentationResult
            At most one mask per prompt, in prompt order.
        """
        best_positions: list[int] = []
        for prompt_index in range(len(self.prompts)):
            positions: IntArray = np.flatnonzero(self.prompt_indices == prompt_index).astype(np.int64)
            if len(positions):
                best_positions.append(int(positions[np.argmax(self.scores[positions])]))
        return self.select(np.array(best_positions, dtype=np.int64))

    def non_maximum_suppression(self, iou_threshold: float) -> "ObjectSegmentationResult":
        """
        Remove overlapping masks by mask IoU, keeping the one with the highest predicted IoU.

        Parameters
        ----------
        iou_threshold : float
            Masks overlapping a kept mask with IoU above this value are removed.

        Raises
        ------
        ValueError
            If ``iou_threshold`` is outside ``[0, 1]``.

        Returns
        -------
        ObjectSegmentationResult
            Kept masks ordered by descending predicted IoU.
        """
        return self.select(MaskSuppression.kept_indices(self.masks, self.pixel_xyxy, self.scores, iou_threshold))

    def union_mask(self) -> BinaryImage:
        """
        Union of every mask.

        Returns
        -------
        BinaryImage
            Pixels of any mask; empty if there is none.
        """
        return BinaryImage(value=self.masks.any(axis=0))

    def instance_map(self) -> IntArray:
        """
        Label map: the mask covering every pixel.

        Where masks overlap, the one with the highest predicted IoU claims the pixel.

        Returns
        -------
        IntArray
            Index of the mask in this result for each pixel, ``-1`` where no mask lies, shape (H, W).
        """
        instance_map: IntArray = np.full((self.image_size.height, self.image_size.width), -1, dtype=np.int64)
        for index in np.argsort(self.scores, kind="stable").tolist():
            instance_map[self.masks[index]] = index
        return instance_map

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(masks={len(self)}, prompts={len(self.prompts)}, "
            + f"image_size=({self.image_size.width}, {self.image_size.height}))"
        )
