from dataclasses import dataclass
from typing import TYPE_CHECKING

from open_vocabulary_detector import Device

from ...options import Precision, TrackerBackend

if TYPE_CHECKING:
    from .tracker import ObjectTracker


@dataclass(frozen=True, kw_only=True)
class ObjectTrackerSettings:
    """
    Everything needed to load and run a video object tracker, identical for every backend.

    The fields map one-to-one onto the ``object_tracker`` section of the YAML presets under ``config/tracker``.

    Attributes
    ----------
    backend : TrackerBackend
        Model family able to load ``weights_path``.
    weights_path : str
        Hugging Face Hub model id or local checkpoint directory.
    device : Device
        Device to run the model on.
    precision : Precision
        Precision of the model weights, frames and tracking state.
    is_frame_storage_on_device : bool
        Whether a whole video given to ``start_session`` is preprocessed and kept on the model device; faster on a
        GPU, but a preprocessed frame takes about 12 MB, so long videos are kept on the CPU by default.
    is_tracking_state_on_device : bool
        Whether the per-frame outputs and memories of an offline session stay on the model device; faster on a GPU,
        but they take about 6 MB per object and frame, so they are kept on the CPU by default.
    stream_conditioning_frame_limit : int
        Maximum number of prompted frames per object a stream keeps in its memory: the first one and the latest
        ones, so re-prompting a stream again and again keeps its footprint bounded.

    Raises
    ------
    ValueError
        If ``weights_path`` is blank or ``stream_conditioning_frame_limit`` is below 2.
    """

    backend: TrackerBackend
    weights_path: str
    device: Device = Device.AUTO
    precision: Precision = Precision.FLOAT32
    is_frame_storage_on_device: bool = False
    is_tracking_state_on_device: bool = False
    stream_conditioning_frame_limit: int = 8

    def __post_init__(self) -> None:
        if not self.weights_path.strip():
            raise ValueError("weights_path must not be blank.")
        if self.stream_conditioning_frame_limit < 2:
            raise ValueError(
                f"stream_conditioning_frame_limit must be at least 2. got {self.stream_conditioning_frame_limit}"
            )

    def build(self) -> "ObjectTracker":
        """
        Load the tracker of ``backend`` with these settings.

        Returns
        -------
        ObjectTracker
            Loaded tracker.

        Raises
        ------
        ValueError
            If the checkpoint has an architecture of another backend.
        """
        match self.backend:
            case TrackerBackend.SAM2 | TrackerBackend.SAM3 | TrackerBackend.EDGETAM:
                from ...backends.memory_tracker import MemoryObjectTracker

                return MemoryObjectTracker(self)
