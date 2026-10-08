from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationThresholds:
    """
    Post-processing thresholds of a segmenter.

    Attributes
    ----------
    confidence_threshold : float
        Minimum confidence for an instance to be kept.
    nms_iou_threshold : float | None
        Mask IoU threshold of class-wise non-maximum suppression; ``None`` disables it.

    Raises
    ------
    ValueError
        If a threshold is outside ``[0, 1]``.
    """

    confidence_threshold: float
    nms_iou_threshold: float | None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1]. got {self.confidence_threshold}")
        if self.nms_iou_threshold is not None and not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError(f"nms_iou_threshold must be in [0, 1]. got {self.nms_iou_threshold}")
