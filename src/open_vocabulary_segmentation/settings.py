from dataclasses import dataclass
from typing import TYPE_CHECKING

from open_vocabulary_detector import DetectorSettings, Device

from .options import Precision, SegmentationThresholds, SegmenterBackend

if TYPE_CHECKING:
    from .segmenter import OpenVocabularySegmenter


@dataclass(frozen=True, kw_only=True)
class SegmenterSettings:
    """
    Everything needed to load and run a segmenter, identical for every backend.

    The fields map one-to-one onto the ``segmenter`` section of the YAML presets in this package, so a preset can be
    turned into settings by any dataclass builder (enums are written by value). ``box_detector`` is a nested
    ``open_vocabulary_detector`` section.

    Attributes
    ----------
    backend : SegmenterBackend
        Segmenter family able to load ``weights_path``.
    weights_path : str
        Hugging Face Hub model id or local checkpoint directory.
    thresholds : SegmentationThresholds
        Post-processing thresholds of the masks.
    box_detector : DetectorSettings | None
        Open-vocabulary detector finding the boxes a box-prompted backend masks; ``None`` for the other backends.
        Its own thresholds select the boxes.
    batch_size : int
        Number of images per forward pass.
    device : Device
        Device to run the segmentation model on; the box detector has its own.
    precision : Precision
        Precision of the segmentation model; ``float16`` is GPU only.

    Raises
    ------
    ValueError
        If ``weights_path`` is blank, ``batch_size`` is not positive, or ``box_detector`` is missing for a
        box-prompted backend or given for another backend.
    """

    backend: SegmenterBackend
    weights_path: str
    thresholds: SegmentationThresholds
    box_detector: DetectorSettings | None = None
    batch_size: int = 4
    device: Device = Device.AUTO
    precision: Precision = Precision.FLOAT32

    def __post_init__(self) -> None:
        if not self.weights_path.strip():
            raise ValueError("weights_path must not be blank.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive. got {self.batch_size}")
        if self.backend.is_box_prompted and self.box_detector is None:
            raise ValueError(f"{self.backend} masks detected boxes and needs box_detector settings.")
        if not self.backend.is_box_prompted and self.box_detector is not None:
            raise ValueError(f"{self.backend} finds instances itself and takes no box_detector settings.")

    def build(self) -> "OpenVocabularySegmenter":
        """
        Load the segmenter of ``backend`` with these settings.

        Only the sub-package of ``backend`` is imported. Box-prompted backends also build their ``box_detector`` and
        the promptable segmenter masking its boxes.

        Returns
        -------
        OpenVocabularySegmenter
            Loaded segmenter.

        Raises
        ------
        ValueError
            If the checkpoint has an architecture of another backend.
        """
        match self.backend:
            case SegmenterBackend.SAM | SegmenterBackend.SAM2 | SegmenterBackend.EDGETAM:
                from .backends.box_prompted import BoxPromptedSegmenter

                return BoxPromptedSegmenter(self)
            case SegmenterBackend.SAM3:
                from .backends.sam3 import Sam3Segmenter

                return Sam3Segmenter(self)
            case SegmenterBackend.FLORENCE2:
                from .backends.florence2 import Florence2Segmenter

                return Florence2Segmenter(self)
            case SegmenterBackend.FLORENCE2_REFERRING:
                from .backends.florence2 import Florence2ReferringSegmenter

                return Florence2ReferringSegmenter(self)
