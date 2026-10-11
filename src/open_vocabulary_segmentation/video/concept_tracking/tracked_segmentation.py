from dataclasses import dataclass

from ...result import Segmentation


@dataclass(frozen=True, eq=False)
class TrackedSegmentation:
    """
    One instance of a tracked concept on one video frame.

    Attributes
    ----------
    track_id : int
        Identifier of the track, stable across the frames of a session.
    frame_index : int
        Index of the frame in the video.
    segmentation : Segmentation
        Mask, box, confidence (detection score that started the track), class and matched query of the instance on
        the frame.
    tracker_score : float
        Presence score of the instance on this frame, in ``[0, 1]``.
    """

    track_id: int
    frame_index: int
    segmentation: Segmentation
    tracker_score: float
