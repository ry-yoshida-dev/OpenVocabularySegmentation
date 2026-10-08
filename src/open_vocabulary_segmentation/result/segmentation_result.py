from collections.abc import Collection, Iterator
from dataclasses import dataclass, field

import numpy as np
from geometry import Box2D, Box2DFormat, Boxes2D
from image_container import BinaryImage
from open_vocabulary_detector import ImageSize, Prompt, PromptQuery

from ..array_types import BoolArray, FloatArray, IntArray
from .mask_overlap import MaskOverlap
from .segmentation import Segmentation


@dataclass(frozen=True, eq=False)
class SegmentationResult:
    """
    Model-independent instance masks for one image.

    Each instance keeps the prompt query that matched it; its class is the class of that query, so several queries
    (e.g. ``"suv"`` and ``"taxi"``) report the same class (``"car"``). Boxes are derived from the masks, so every
    backend reports them the same way.

    Attributes
    ----------
    masks : BoolArray
        Pixels of each instance in the source image, shape (N, H, W).
    confidences : FloatArray
        Confidence of each instance in ``[0, 1]``, shape (N,).
    query_ids : IntArray
        Matched query of each instance, indexing ``prompt.queries``, shape (N,).
    prompt : Prompt
        Prompt whose queries were segmented.
    image_size : ImageSize
        Size of the image the masks cover.
    pixel_xyxy : IntArray
        Bounding box of each mask as pixel bounds with exclusive right and bottom edges, shape (N, 4).

    Raises
    ------
    ValueError
        If the masks do not match the image size, a mask is empty, array lengths disagree, or a query id is out of
        range.
    """

    masks: BoolArray
    confidences: FloatArray
    query_ids: IntArray
    prompt: Prompt
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
        instance_count: int = int(self.masks.shape[0])
        if self.confidences.shape != (instance_count,):
            raise ValueError(f"confidences must have shape ({instance_count},). got {self.confidences.shape}")
        if self.query_ids.shape != (instance_count,):
            raise ValueError(f"query_ids must have shape ({instance_count},). got {self.query_ids.shape}")
        query_count: int = len(self.prompt.queries)
        if instance_count and (self.query_ids.min() < 0 or self.query_ids.max() >= query_count):
            raise ValueError(f"query_ids must be in [0, {query_count}). got {self.query_ids}")
        is_column_filled: BoolArray = self.masks.any(axis=1)
        is_row_filled: BoolArray = self.masks.any(axis=2)
        if not bool(is_column_filled.any(axis=1).all()):
            raise ValueError("masks must not be empty; build the result with from_masks to drop empty masks.")
        object.__setattr__(
            self,
            "pixel_xyxy",
            np.stack(
                [
                    is_column_filled.argmax(axis=1),
                    is_row_filled.argmax(axis=1),
                    self.image_size.width - is_column_filled[:, ::-1].argmax(axis=1),
                    self.image_size.height - is_row_filled[:, ::-1].argmax(axis=1),
                ],
                axis=1,
            )
            .astype(np.int64)
            .reshape(-1, 4),
        )

    @classmethod
    def from_masks(
        cls,
        masks: BoolArray,
        confidences: FloatArray,
        query_ids: IntArray,
        prompt: Prompt,
        image_size: ImageSize,
    ) -> "SegmentationResult":
        """
        Build a result from raw masks, dropping empty masks together with their confidences and query ids.

        Parameters
        ----------
        masks : BoolArray
            Pixels of each instance in the source image, shape (N, H, W).
        confidences : FloatArray
            Confidence of each instance in ``[0, 1]``, shape (N,).
        query_ids : IntArray
            Matched query of each instance, indexing ``prompt.queries``, shape (N,).
        prompt : Prompt
            Prompt whose queries were segmented.
        image_size : ImageSize
            Size of the source image.

        Returns
        -------
        SegmentationResult
            Instances with at least one pixel.
        """
        boolean_masks: BoolArray = np.asarray(masks, dtype=np.bool_).reshape(-1, image_size.height, image_size.width)
        is_filled: BoolArray = boolean_masks.any(axis=(1, 2))
        return cls(
            masks=boolean_masks[is_filled],
            confidences=np.asarray(confidences, dtype=np.float64).reshape(-1)[is_filled],
            query_ids=np.asarray(query_ids, dtype=np.int64).reshape(-1)[is_filled],
            prompt=prompt,
            image_size=image_size,
        )

    @classmethod
    def empty(cls, prompt: Prompt, image_size: ImageSize) -> "SegmentationResult":
        """
        Build a result without instances.

        Parameters
        ----------
        prompt : Prompt
            Prompt whose queries were segmented.
        image_size : ImageSize
            Size of the source image.

        Returns
        -------
        SegmentationResult
            Result with zero instances.
        """
        return cls(
            masks=np.zeros((0, image_size.height, image_size.width), dtype=np.bool_),
            confidences=np.zeros((0,), dtype=np.float64),
            query_ids=np.zeros((0,), dtype=np.int64),
            prompt=prompt,
            image_size=image_size,
        )

    @property
    def boxes(self) -> Boxes2D:
        """
        Bounding box of every mask.

        Returns
        -------
        Boxes2D
            Absolute XYXY pixel boxes enclosing every pixel of each mask, shape (N, 4).
        """
        return Boxes2D.register(value=self.xyxy, box2d_format=Box2DFormat.XYXY)

    @property
    def xyxy(self) -> FloatArray:
        """
        Bounding boxes as a plain array.

        Returns
        -------
        FloatArray
            Absolute XYXY pixel boxes enclosing every pixel of each mask, shape (N, 4).
        """
        return self.pixel_xyxy.astype(np.float64)

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
    def class_names(self) -> tuple[str, ...]:
        """
        Output class names of the prompt.

        Returns
        -------
        tuple[str, ...]
            Names in class-id order.
        """
        return self.prompt.class_names

    @property
    def class_ids(self) -> IntArray:
        """
        Output class of each instance, i.e. the class of its matched query.

        Returns
        -------
        IntArray
            Indices into ``class_names``, shape (N,).
        """
        return self.prompt.class_ids_of(self.query_ids)

    def __len__(self) -> int:
        return int(self.masks.shape[0])

    def __iter__(self) -> Iterator[Segmentation]:
        xyxy: FloatArray = self.xyxy
        for index in range(len(self)):
            query_id: int = int(self.query_ids[index])
            class_id: int = self.prompt.query_class_ids[query_id]
            yield Segmentation(
                mask=self.masks[index],
                box=Box2D.register(value=xyxy[index], box2d_format=Box2DFormat.XYXY),
                confidence=float(self.confidences[index]),
                class_id=class_id,
                class_name=self.class_names[class_id],
                matched_query=self.prompt.queries[query_id],
            )

    def select(self, indices: BoolArray | IntArray) -> "SegmentationResult":
        """
        Keep a subset of instances.

        Parameters
        ----------
        indices : BoolArray | IntArray
            Boolean mask of shape (N,) or integer indices.

        Returns
        -------
        SegmentationResult
            Instances at the given indices, in the given order.
        """
        return SegmentationResult(
            masks=self.masks[indices].reshape(-1, self.image_size.height, self.image_size.width),
            confidences=self.confidences[indices].reshape(-1),
            query_ids=self.query_ids[indices].reshape(-1),
            prompt=self.prompt,
            image_size=self.image_size,
        )

    def filter_by_confidence(self, confidence_threshold: float) -> "SegmentationResult":
        """
        Keep instances whose confidence is at least ``confidence_threshold``.

        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence to keep.

        Returns
        -------
        SegmentationResult
            Filtered instances.
        """
        return self.select(self.confidences >= confidence_threshold)

    def sort_by_confidence(self) -> "SegmentationResult":
        """
        Order instances by descending confidence.

        Returns
        -------
        SegmentationResult
            Sorted instances.
        """
        order: IntArray = np.argsort(-self.confidences, kind="stable").astype(np.int64)
        return self.select(order)

    def top_k(self, count: int) -> "SegmentationResult":
        """
        Keep the most confident instances.

        Parameters
        ----------
        count : int
            Maximum number of instances to keep.

        Raises
        ------
        ValueError
            If ``count`` is negative.

        Returns
        -------
        SegmentationResult
            At most ``count`` instances ordered by descending confidence.
        """
        if count < 0:
            raise ValueError(f"count must not be negative. got {count}")
        return self.sort_by_confidence().select(np.arange(min(count, len(self)), dtype=np.int64))

    def filter_by_class(self, class_name: str) -> "SegmentationResult":
        """
        Keep the instances of one class.

        Parameters
        ----------
        class_name : str
            Output class name of the prompt.

        Raises
        ------
        KeyError
            If the prompt has no class ``class_name``.

        Returns
        -------
        SegmentationResult
            Instances of ``class_name``, in the current order.
        """
        return self.filter_by_classes((class_name,))

    def filter_by_classes(self, class_names: Collection[str]) -> "SegmentationResult":
        """
        Keep the instances of several classes.

        Parameters
        ----------
        class_names : Collection[str]
            Output class names of the prompt.

        Raises
        ------
        KeyError
            If the prompt has no class of one of ``class_names``.

        Returns
        -------
        SegmentationResult
            Instances of ``class_names``, in the current order.
        """
        class_ids: list[int] = [self.class_id_of(class_name) for class_name in class_names]
        return self.select(np.isin(self.class_ids, class_ids))

    def filter_by_query(self, query: PromptQuery) -> "SegmentationResult":
        """
        Keep the instances matched by one query.

        Parameters
        ----------
        query : PromptQuery
            Query of the prompt, e.g. ``TextQuery("taxi")``.

        Raises
        ------
        KeyError
            If ``query`` is not a query of the prompt.

        Returns
        -------
        SegmentationResult
            Instances matched by ``query``, in the current order.
        """
        if query not in self.prompt.queries:
            raise KeyError(f"{query!r} is not a query of the prompt.")
        return self.select(self.query_ids == self.prompt.queries.index(query))

    def filter_by_area(self, minimum_area: int, maximum_area: int | None = None) -> "SegmentationResult":
        """
        Keep the instances whose pixel count lies in a range, e.g. to drop specks.

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
        SegmentationResult
            Instances within the range, in the current order.
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

    def non_maximum_suppression(self, iou_threshold: float, is_class_agnostic: bool = False) -> "SegmentationResult":
        """
        Remove overlapping instances by mask IoU, keeping the most confident one.

        Parameters
        ----------
        iou_threshold : float
            Instances whose mask overlaps a kept mask with IoU above this value are removed.
        is_class_agnostic : bool, optional
            If True, suppress across classes; otherwise only within each class, so instances matched by different
            queries of one class suppress each other.

        Raises
        ------
        ValueError
            If ``iou_threshold`` is outside ``[0, 1]``.

        Returns
        -------
        SegmentationResult
            Kept instances ordered by descending confidence.
        """
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1]. got {iou_threshold}")
        if not len(self):
            return self
        overlaps: FloatArray = MaskOverlap.pairwise_iou(self.masks, self.pixel_xyxy)
        if not is_class_agnostic:
            class_ids: IntArray = self.class_ids
            is_same_class: BoolArray = class_ids[:, None] == class_ids[None, :]
            overlaps = np.where(is_same_class, overlaps, 0.0)
        is_suppressed: BoolArray = np.zeros(len(self), dtype=np.bool_)
        kept_indices: list[int] = []
        for index in np.argsort(-self.confidences, kind="stable").tolist():
            if is_suppressed[index]:
                continue
            kept_indices.append(index)
            is_suppressed |= overlaps[index] > iou_threshold
        return self.select(np.array(kept_indices, dtype=np.int64))

    def class_map(self) -> IntArray:
        """
        Semantic segmentation map: the class of every pixel.

        Where instances overlap, the most confident one decides the class.

        Returns
        -------
        IntArray
            Class id of each pixel, ``-1`` where no instance lies, shape (H, W).
        """
        class_map: IntArray = np.full((self.image_size.height, self.image_size.width), -1, dtype=np.int64)
        class_ids: IntArray = self.class_ids
        for index in np.argsort(self.confidences, kind="stable").tolist():
            class_map[self.masks[index]] = class_ids[index]
        return class_map

    def instance_map(self) -> IntArray:
        """
        Instance label map: the instance covering every pixel.

        Where instances overlap, the most confident one claims the pixel.

        Returns
        -------
        IntArray
            Index of the instance in this result for each pixel, ``-1`` where no instance lies, shape (H, W).
        """
        instance_map: IntArray = np.full((self.image_size.height, self.image_size.width), -1, dtype=np.int64)
        for index in np.argsort(self.confidences, kind="stable").tolist():
            instance_map[self.masks[index]] = index
        return instance_map

    def class_mask(self, class_name: str) -> BinaryImage:
        """
        Union of the masks of every instance of one class.

        Overlaps with other classes are kept, unlike ``class_map``.

        Parameters
        ----------
        class_name : str
            Output class name of the prompt.

        Raises
        ------
        KeyError
            If the prompt has no class ``class_name``.

        Returns
        -------
        BinaryImage
            Pixels of any instance of ``class_name``; empty if there is none.
        """
        return self.filter_by_class(class_name).union_mask()

    def class_masks(self) -> dict[str, BinaryImage]:
        """
        Union mask of every class of the prompt.

        Returns
        -------
        dict[str, BinaryImage]
            Mask of each class keyed by class name, in class-id order; classes without instances map to empty masks.
        """
        return {class_name: self.class_mask(class_name) for class_name in self.class_names}

    def union_mask(self) -> BinaryImage:
        """
        Union of the masks of every instance, i.e. the foreground.

        Returns
        -------
        BinaryImage
            Pixels of any instance; empty if there is none.
        """
        return BinaryImage(value=self.masks.any(axis=0))

    def class_counts(self) -> dict[str, int]:
        """
        Number of instances of every class of the prompt.

        Returns
        -------
        dict[str, int]
            Instance count keyed by class name, in class-id order, zero for classes without instances.
        """
        counts: IntArray = np.bincount(self.class_ids, minlength=len(self.class_names)).astype(np.int64)
        return dict(zip(self.class_names, counts.tolist(), strict=True))

    def class_id_of(self, class_name: str) -> int:
        """
        Class id of a class name of the prompt.

        Parameters
        ----------
        class_name : str
            Output class name; surrounding whitespace is ignored like in ``Prompt``.

        Raises
        ------
        KeyError
            If the prompt has no class ``class_name``.

        Returns
        -------
        int
            Index into ``class_names``.
        """
        stripped_name: str = class_name.strip()
        if stripped_name not in self.class_names:
            raise KeyError(f"{class_name!r} is not a class of the prompt. classes: {list(self.class_names)}")
        return self.class_names.index(stripped_name)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(instances={len(self)}, "
            + f"image_size=({self.image_size.width}, {self.image_size.height}), "
            + f"class_names={self.class_names})"
        )
