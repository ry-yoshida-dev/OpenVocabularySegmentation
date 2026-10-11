from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ConceptTrackingParameters:
    """
    Overrides of the detection-tracking fusion of SAM 3 video; ``None`` keeps the value of the checkpoint.

    Defaults in brackets are those of ``facebook/sam3``.

    Attributes
    ----------
    detection_threshold : float | None
        Minimum score of a detection to be matched with tracks (``score_threshold_detection``, 0.5).
    new_track_threshold : float | None
        Minimum score of an unmatched detection to start a new track (``new_det_thresh``, 0.7).
    detection_nms_iou_threshold : float | None
        IoU above which overlapping detections of a frame are suppressed (``det_nms_thresh``, 0.1).
    association_iou_threshold : float | None
        IoU above which a detection is matched with a track (``assoc_iou_thresh``, 0.1).
    track_association_iou_threshold : float | None
        IoU a track needs with a detection not to count as unmatched (``trk_assoc_iou_thresh``, 0.5).
    hotstart_delay : int | None
        Frames a new track is held back while duplicate and spurious tracks are removed (``hotstart_delay``, 15);
        ``propagate`` yields outputs this many frames late, ``0`` disables the delay and the removal.
    hotstart_unmatched_frame_count : int | None
        Unmatched frames after which a new track is removed during the hot start (``hotstart_unmatch_thresh``, 8);
        must not exceed the hot start delay.
    hotstart_duplicate_frame_count : int | None
        Frames of overlap with an older track after which a new track is removed as a duplicate during the hot start
        (``hotstart_dup_thresh``, 8); must not exceed the hot start delay.
    recondition_interval : int | None
        Every this many frames, tracks matched by a confident detection are re-anchored on it
        (``recondition_every_nth_frame``, 16); ``0`` disables reconditioning.
    recondition_confidence_threshold : float | None
        Minimum detection score to recondition a track (``high_conf_thresh``, 0.8).
    recondition_iou_threshold : float | None
        Minimum IoU between the detection and the track to recondition it (``high_iou_thresh``, 0.8).
    occlusion_suppression_iou_threshold : float | None
        IoU above which the most recently occluded of two overlapping tracks is hidden
        (``suppress_overlapping_based_on_recent_occlusion_threshold``, 0.7).
    max_object_count : int | None
        Maximum number of tracks (``max_num_objects``).
    fill_hole_area : int | None
        Holes and specks up to this many pixels are removed from masks (``fill_hole_area``, 16); needs the
        ``kernels`` package.

    Raises
    ------
    ValueError
        If a threshold is outside ``[0, 1]``, a count is negative, or a hot start frame count exceeds the hot start
        delay.
    """

    detection_threshold: float | None = None
    new_track_threshold: float | None = None
    detection_nms_iou_threshold: float | None = None
    association_iou_threshold: float | None = None
    track_association_iou_threshold: float | None = None
    hotstart_delay: int | None = None
    hotstart_unmatched_frame_count: int | None = None
    hotstart_duplicate_frame_count: int | None = None
    recondition_interval: int | None = None
    recondition_confidence_threshold: float | None = None
    recondition_iou_threshold: float | None = None
    occlusion_suppression_iou_threshold: float | None = None
    max_object_count: int | None = None
    fill_hole_area: int | None = None

    def __post_init__(self) -> None:
        for name, threshold in (
            ("detection_threshold", self.detection_threshold),
            ("new_track_threshold", self.new_track_threshold),
            ("detection_nms_iou_threshold", self.detection_nms_iou_threshold),
            ("association_iou_threshold", self.association_iou_threshold),
            ("track_association_iou_threshold", self.track_association_iou_threshold),
            ("recondition_confidence_threshold", self.recondition_confidence_threshold),
            ("recondition_iou_threshold", self.recondition_iou_threshold),
            ("occlusion_suppression_iou_threshold", self.occlusion_suppression_iou_threshold),
        ):
            if threshold is not None and not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]. got {threshold}")
        for name, count in (
            ("hotstart_delay", self.hotstart_delay),
            ("hotstart_unmatched_frame_count", self.hotstart_unmatched_frame_count),
            ("hotstart_duplicate_frame_count", self.hotstart_duplicate_frame_count),
            ("recondition_interval", self.recondition_interval),
            ("max_object_count", self.max_object_count),
            ("fill_hole_area", self.fill_hole_area),
        ):
            if count is not None and count < 0:
                raise ValueError(f"{name} must not be negative. got {count}")
        if self.hotstart_delay:
            for name, count in (
                ("hotstart_unmatched_frame_count", self.hotstart_unmatched_frame_count),
                ("hotstart_duplicate_frame_count", self.hotstart_duplicate_frame_count),
            ):
                if count is not None and count > self.hotstart_delay:
                    raise ValueError(f"{name} must not exceed hotstart_delay ({self.hotstart_delay}). got {count}")
