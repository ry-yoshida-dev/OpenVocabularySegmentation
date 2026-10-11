from typing import ClassVar

from transformers import EdgeTamModel, Sam2Model

from ...options import PromptableBackend
from ..sam2 import Sam2PromptableSegmenter


class EdgeTamPromptableSegmenter(Sam2PromptableSegmenter):
    """
    EdgeTAM backed by ``transformers.EdgeTamModel``.

    EdgeTAM is SAM 2 with a RepViT image encoder (``timm``) small enough for on-device use; its processor, multi-scale
    embedding and mask decoder are those of SAM 2. The published checkpoint is a video checkpoint (``edgetam_video``)
    whose image part is loaded.
    """

    BACKEND: ClassVar[PromptableBackend] = PromptableBackend.EDGETAM

    def _load_model(self, weights_path: str) -> Sam2Model | EdgeTamModel:
        """
        Load the EdgeTAM image model of the checkpoint.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        Sam2Model | EdgeTamModel
            EdgeTAM image model.
        """
        return EdgeTamModel.from_pretrained(weights_path)
