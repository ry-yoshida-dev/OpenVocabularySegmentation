from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from geometry import Box2D, Box2DFormat
from image_container import BinaryImage
from open_vocabulary_detector import ImageSize

from ...array_types import BoolArray, FloatArray, IntArray
from ...result import MaskBounds
from .tracked_object import TrackedObject


@dataclass(frozen=True, eq=False)
class TrackedFrame:
    """
    Masks of every tracked object on one video frame.

    Attributes
    ----------
    frame_index : int
        Index of the frame in the video.
    object_ids : IntArray
        Identifiers of the tracked objects, shape (N,).
    masks : BoolArray
        Pixels of each object on the frame, shape (N, H, W); empty for occluded objects.
    presence_scores : FloatArray
        Probability that each object is visible, in ``[0, 1]``, shape (N,).
    image_size : ImageSize
        Size of the frame.
    pixel_xyxy : IntArray
        Bounding box of each mask as pixel bounds with exclusive right and bottom edges, all zeros for an empty
        mask, shape (N, 4).

    Raises
    ------
    ValueError
        If the frame index is negative, the masks do not match the frame size, array lengths disagree, object ids
        repeat or presence scores lie outside ``[0, 1]``.
    """

    frame_index: int
    object_ids: IntArray
    masks: BoolArray
    presence_scores: FloatArray
    image_size: ImageSize
    pixel_xyxy: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index must not be negative. got {self.frame_index}")
        expected_shape: tuple[int, int] = (self.image_size.height, self.image_size.width)
        if self.masks.ndim != 3 or self.masks.shape[1:] != expected_shape:
            raise ValueError(
                f"masks must have shape (N, {expected_shape[0]}, {expected_shape[1]}). got {self.masks.shape}"
            )
        object_count: int = int(self.masks.shape[0])
        if self.object_ids.shape != (object_count,) or self.presence_scores.shape != (object_count,):
            raise ValueError(
                f"object_ids and presence_scores must have shape ({object_count},). "
                + f"got {self.object_ids.shape} and {self.presence_scores.shape}"
            )
        if len(np.unique(self.object_ids)) != object_count:
            raise ValueError(f"object_ids must be unique. got {self.object_ids.tolist()}")
        if bool(((self.presence_scores < 0.0) | (self.presence_scores > 1.0)).any()):
            raise ValueError(f"presence_scores must be in [0, 1]. got {self.presence_scores.tolist()}")
        object.__setattr__(self, "pixel_xyxy", MaskBounds.pixel_xyxy(self.masks))

    @classmethod
    def empty(cls, frame_index: int, image_size: ImageSize) -> "TrackedFrame":
        """
        Build a frame without tracked objects.

        Parameters
        ----------
        frame_index : int
            Index of the frame in the video.
        image_size : ImageSize
            Size of the frame.

        Returns
        -------
        TrackedFrame
            Frame with zero objects.
        """
        return cls(
            frame_index=frame_index,
            object_ids=np.zeros((0,), dtype=np.int64),
            masks=np.zeros((0, image_size.height, image_size.width), dtype=np.bool_),
            presence_scores=np.zeros((0,), dtype=np.float64),
            image_size=image_size,
        )

    def __len__(self) -> int:
        return int(self.masks.shape[0])

    def __iter__(self) -> Iterator[TrackedObject]:
        for index in range(len(self)):
            yield self._object_at(index)

    def __contains__(self, object_id: int) -> bool:
        return bool((self.object_ids == object_id).any())

    def object(self, object_id: int) -> TrackedObject:
        """
        One tracked object.

        Parameters
        ----------
        object_id : int
            Identifier the object was prompted with.

        Raises
        ------
        KeyError
            If no object has ``object_id``.

        Returns
        -------
        TrackedObject
            Mask and presence of the object on this frame.
        """
        positions: IntArray = np.flatnonzero(self.object_ids == object_id).astype(np.int64)
        if not len(positions):
            raise KeyError(f"object {object_id} is not tracked. tracked: {self.object_ids.tolist()}")
        return self._object_at(int(positions[0]))

    def visible_objects(self, minimum_presence_score: float = 0.5) -> "TrackedFrame":
        """
        Keep the objects that are visible on the frame.

        Parameters
        ----------
        minimum_presence_score : float, optional
            Minimum probability of being visible.

        Returns
        -------
        TrackedFrame
            Objects with a non-empty mask and a presence score of at least ``minimum_presence_score``.
        """
        is_kept: BoolArray = self.masks.any(axis=(1, 2)) & (self.presence_scores >= minimum_presence_score)
        return TrackedFrame(
            frame_index=self.frame_index,
            object_ids=self.object_ids[is_kept],
            masks=self.masks[is_kept],
            presence_scores=self.presence_scores[is_kept],
            image_size=self.image_size,
        )

    def union_mask(self) -> BinaryImage:
        """
        Union of every object mask.

        Returns
        -------
        BinaryImage
            Pixels of any object; empty if there is none.
        """
        return BinaryImage(value=self.masks.any(axis=0))

    def object_map(self) -> IntArray:
        """
        Label map of object identifiers.

        Where masks overlap, the object prompted later claims the pixel.

        Returns
        -------
        IntArray
            Object id of each pixel, ``-1`` where no object lies, shape (H, W).
        """
        object_map: IntArray = np.full((self.image_size.height, self.image_size.width), -1, dtype=np.int64)
        for index in range(len(self)):
            object_map[self.masks[index]] = int(self.object_ids[index])
        return object_map

    def _object_at(self, index: int) -> TrackedObject:
        is_empty: bool = not bool(self.masks[index].any())
        return TrackedObject(
            object_id=int(self.object_ids[index]),
            frame_index=self.frame_index,
            mask=self.masks[index],
            box=(
                None
                if is_empty
                else Box2D.register(value=self.pixel_xyxy[index].astype(np.float64), box2d_format=Box2DFormat.XYXY)
            ),
            presence_score=float(self.presence_scores[index]),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(frame_index={self.frame_index}, object_ids={self.object_ids.tolist()}, "
            + f"image_size=({self.image_size.width}, {self.image_size.height}))"
        )
