from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from open_vocabulary_detector import ImageSize, Prompt

from ...array_types import FloatArray, IntArray
from ...result import Segmentation, SegmentationResult
from .tracked_segmentation import TrackedSegmentation


@dataclass(frozen=True, eq=False)
class ConceptFrame:
    """
    Instances of the tracked concepts on one video frame, each with its track identifier.

    ``instances`` is a regular ``SegmentationResult`` for the per-frame tools (class masks, label maps, crops); its
    confidences are the detection scores that started each track. ``track_ids`` and ``tracker_scores`` stay aligned
    with it, so select instances through this class to keep their identities.

    Attributes
    ----------
    frame_index : int
        Index of the frame in the video.
    instances : SegmentationResult
        Instances on the frame, in ascending track order.
    track_ids : IntArray
        Track of each instance, shape (N,).
    tracker_scores : FloatArray
        Presence score of each instance on this frame, in ``[0, 1]``, shape (N,).

    Raises
    ------
    ValueError
        If the frame index is negative, the track ids or tracker scores do not match the instances, or the track ids
        repeat.
    """

    frame_index: int
    instances: SegmentationResult
    track_ids: IntArray
    tracker_scores: FloatArray

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index must not be negative. got {self.frame_index}")
        if self.track_ids.shape != (len(self.instances),):
            raise ValueError(f"track_ids must have shape ({len(self.instances)},). got {self.track_ids.shape}")
        if self.tracker_scores.shape != (len(self.instances),):
            raise ValueError(
                f"tracker_scores must have shape ({len(self.instances)},). got {self.tracker_scores.shape}"
            )
        if len(np.unique(self.track_ids)) != len(self.track_ids):
            raise ValueError(f"track_ids must be unique. got {self.track_ids.tolist()}")

    @classmethod
    def empty(cls, frame_index: int, prompt: Prompt, image_size: ImageSize) -> "ConceptFrame":
        """
        Build a frame without instances.

        Parameters
        ----------
        frame_index : int
            Index of the frame in the video.
        prompt : Prompt
            Tracked prompt.
        image_size : ImageSize
            Size of the frame.

        Returns
        -------
        ConceptFrame
            Frame with zero instances.
        """
        return cls(
            frame_index=frame_index,
            instances=SegmentationResult.empty(prompt, image_size),
            track_ids=np.zeros((0,), dtype=np.int64),
            tracker_scores=np.zeros((0,), dtype=np.float64),
        )

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self) -> Iterator[TrackedSegmentation]:
        for track_id, tracker_score, segmentation in zip(
            self.track_ids.tolist(), self.tracker_scores.tolist(), self.instances, strict=True
        ):
            yield TrackedSegmentation(
                track_id=int(track_id),
                frame_index=self.frame_index,
                segmentation=segmentation,
                tracker_score=float(tracker_score),
            )

    def __contains__(self, track_id: int) -> bool:
        return bool((self.track_ids == track_id).any())

    def segmentation_of(self, track_id: int) -> Segmentation:
        """
        Instance of one track on the frame.

        Parameters
        ----------
        track_id : int
            Track identifier.

        Raises
        ------
        KeyError
            If the track has no instance on the frame.

        Returns
        -------
        Segmentation
            Mask, box, confidence and class of the track's instance.
        """
        positions: IntArray = np.flatnonzero(self.track_ids == track_id).astype(np.int64)
        if not len(positions):
            raise KeyError(f"track {track_id} is not on frame {self.frame_index}. tracks: {self.track_ids.tolist()}")
        return next(iter(self.instances.select(positions)))

    def filter_by_class(self, class_name: str) -> "ConceptFrame":
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
        ConceptFrame
            Instances of ``class_name`` with their tracks.
        """
        is_kept = self.instances.class_ids == self.instances.class_id_of(class_name)
        return ConceptFrame(
            frame_index=self.frame_index,
            instances=self.instances.select(is_kept),
            track_ids=self.track_ids[is_kept],
            tracker_scores=self.tracker_scores[is_kept],
        )

    def track_map(self) -> IntArray:
        """
        Label map of track identifiers.

        Where instances overlap, the most confident one claims the pixel.

        Returns
        -------
        IntArray
            Track id of each pixel, ``-1`` where no instance lies, shape (H, W).
        """
        instance_map: IntArray = self.instances.instance_map()
        lookup: IntArray = np.concatenate([self.track_ids, np.array([-1], dtype=np.int64)])
        return lookup[instance_map].astype(np.int64)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(frame_index={self.frame_index}, track_ids={self.track_ids.tolist()}, "
            + f"class_names={self.instances.class_names})"
        )
