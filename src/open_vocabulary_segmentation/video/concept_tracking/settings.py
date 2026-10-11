from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from open_vocabulary_detector import Device

from ...options import Precision
from .parameters import ConceptTrackingParameters

if TYPE_CHECKING:
    from .tracker import ConceptTracker


@dataclass(frozen=True, kw_only=True)
class ConceptTrackerSettings:
    """
    Everything needed to load and run a video concept tracker (SAM 3 video).

    The fields map one-to-one onto the ``concept_tracker`` section of the YAML presets under
    ``config/concept_tracker``.

    Attributes
    ----------
    weights_path : str
        Hugging Face Hub model id or local checkpoint directory of a ``sam3_video`` checkpoint.
    confidence_threshold : float
        Minimum detection score of a reported instance; SAM 3 scores a track once, by the detection that started it.
    parameters : ConceptTrackingParameters
        Overrides of the detection-tracking fusion.
    device : Device
        Device to run the model on.
    precision : Precision
        Precision of the model weights, frames and tracking state.
    tracker_score_threshold : float
        Minimum per-frame tracker score (presence of the object on the frame) of a reported instance.
    is_frame_storage_on_device : bool
        Whether a whole video given to ``start_session`` is preprocessed and kept on the model device.
    is_tracking_state_on_device : bool
        Whether the per-frame outputs and memories of an offline session stay on the model device instead of the CPU.
    stream_conditioning_frame_limit : int
        Maximum number of conditioning frames per track a stream keeps in its memory: the first one and the latest
        ones.
    stream_lost_track_frame_count : int
        A stream drops a track that has had no visible instance for this many frames, so the tracks of a long stream
        stay bounded; SAM 3 has no hot start removal in streams.

    Raises
    ------
    ValueError
        If ``weights_path`` is blank, a threshold is outside ``[0, 1]``, ``stream_conditioning_frame_limit`` is below 2
        or ``stream_lost_track_frame_count`` is not positive.
    """

    weights_path: str
    confidence_threshold: float = 0.0
    parameters: ConceptTrackingParameters = field(default_factory=ConceptTrackingParameters)
    device: Device = Device.AUTO
    precision: Precision = Precision.FLOAT32
    tracker_score_threshold: float = 0.0
    is_frame_storage_on_device: bool = False
    is_tracking_state_on_device: bool = False
    stream_conditioning_frame_limit: int = 8
    stream_lost_track_frame_count: int = 300

    def __post_init__(self) -> None:
        if not self.weights_path.strip():
            raise ValueError("weights_path must not be blank.")
        for name, threshold in (
            ("confidence_threshold", self.confidence_threshold),
            ("tracker_score_threshold", self.tracker_score_threshold),
        ):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]. got {threshold}")
        if self.stream_conditioning_frame_limit < 2:
            raise ValueError(
                f"stream_conditioning_frame_limit must be at least 2. got {self.stream_conditioning_frame_limit}"
            )
        if self.stream_lost_track_frame_count <= 0:
            raise ValueError(
                f"stream_lost_track_frame_count must be positive. got {self.stream_lost_track_frame_count}"
            )

    def build(self) -> "ConceptTracker":
        """
        Load the concept tracker with these settings.

        Returns
        -------
        ConceptTracker
            Loaded tracker.

        Raises
        ------
        ValueError
            If the checkpoint is not a SAM 3 video checkpoint.
        """
        from ...backends.sam3 import Sam3ConceptTracker

        return Sam3ConceptTracker(self)
