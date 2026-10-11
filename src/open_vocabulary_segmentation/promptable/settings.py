from dataclasses import dataclass
from typing import TYPE_CHECKING

from open_vocabulary_detector import Device

from ..options import Precision, PromptableBackend

if TYPE_CHECKING:
    from .segmenter import PromptableSegmenter


@dataclass(frozen=True, kw_only=True)
class PromptableSegmenterSettings:
    """
    Everything needed to load and run a promptable segmenter, identical for every backend.

    The fields map one-to-one onto the ``promptable_segmenter`` section of the YAML presets under
    ``config/promptable``.

    Attributes
    ----------
    backend : PromptableBackend
        Model family able to load ``weights_path``.
    weights_path : str
        Hugging Face Hub model id or local checkpoint directory.
    batch_size : int
        Number of images encoded per forward pass.
    decode_batch_size : int
        Maximum number of object prompts decoded per forward pass.
    device : Device
        Device to run the model on.
    precision : Precision
        Precision of the model weights and inputs.

    Raises
    ------
    ValueError
        If ``weights_path`` is blank or a batch size is not positive.
    """

    backend: PromptableBackend
    weights_path: str
    batch_size: int = 4
    decode_batch_size: int = 64
    device: Device = Device.AUTO
    precision: Precision = Precision.FLOAT32

    def __post_init__(self) -> None:
        if not self.weights_path.strip():
            raise ValueError("weights_path must not be blank.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive. got {self.batch_size}")
        if self.decode_batch_size <= 0:
            raise ValueError(f"decode_batch_size must be positive. got {self.decode_batch_size}")

    def build(self) -> "PromptableSegmenter":
        """
        Load the promptable segmenter of ``backend`` with these settings.

        Only the sub-package of ``backend`` is imported.

        Returns
        -------
        PromptableSegmenter
            Loaded segmenter.

        Raises
        ------
        ValueError
            If the checkpoint has an architecture of another backend.
        """
        match self.backend:
            case PromptableBackend.SAM:
                from ..backends.sam import SamPromptableSegmenter

                return SamPromptableSegmenter(self)
            case PromptableBackend.SAM2:
                from ..backends.sam2 import Sam2PromptableSegmenter

                return Sam2PromptableSegmenter(self)
            case PromptableBackend.SAM3:
                from ..backends.sam3 import Sam3PromptableSegmenter

                return Sam3PromptableSegmenter(self)
            case PromptableBackend.EDGETAM:
                from ..backends.edgetam import EdgeTamPromptableSegmenter

                return EdgeTamPromptableSegmenter(self)
            case PromptableBackend.FLORENCE2:
                from ..backends.florence2 import Florence2PromptableSegmenter

                return Florence2PromptableSegmenter(self)
