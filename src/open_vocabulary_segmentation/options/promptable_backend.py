from enum import StrEnum

from .prompt_element import PromptElement


class PromptableBackend(StrEnum):
    """
    Model family that segments objects given by points, boxes and masks (promptable visual segmentation).

    Attributes
    ----------
    SAM : str
        Segment Anything through Hugging Face ``transformers`` (``SamModel``).
    SAM2 : str
        SAM 2 and SAM 2.1 through Hugging Face ``transformers`` (``Sam2Model``).
    SAM3 : str
        The interactive tracker of SAM 3 through Hugging Face ``transformers`` (``Sam3TrackerModel``), loaded from
        the same checkpoint as SAM 3 concept segmentation.
    EDGETAM : str
        EdgeTAM through Hugging Face ``transformers`` (``EdgeTamModel``): SAM 2 with a light RepViT image encoder for
        on-device use; needs the ``timm`` package.
    FLORENCE2 : str
        Florence-2 region segmentation through Hugging Face ``transformers``; boxes only, without scores or logits.
    """

    SAM = "sam"
    SAM2 = "sam2"
    SAM3 = "sam3"
    EDGETAM = "edgetam"
    FLORENCE2 = "florence2"

    @property
    def model_types(self) -> frozenset[str]:
        """
        ``model_type`` values of the checkpoints the family loads.

        Returns
        -------
        frozenset[str]
            Accepted ``model_type`` of the checkpoint's ``config.json``.
        """
        match self:
            case PromptableBackend.SAM:
                return frozenset({"sam"})
            case PromptableBackend.SAM2:
                return frozenset({"sam2", "sam2_video"})
            case PromptableBackend.SAM3:
                return frozenset({"sam3_video", "sam3_tracker", "sam3_tracker_video"})
            case PromptableBackend.EDGETAM:
                return frozenset({"edgetam", "edgetam_video"})
            case PromptableBackend.FLORENCE2:
                return frozenset({"florence2"})

    @property
    def supported_prompt_elements(self) -> frozenset[PromptElement]:
        """
        Prompt elements the family accepts.

        Returns
        -------
        frozenset[PromptElement]
            Points, boxes and masks for the SAM families, boxes for Florence-2.
        """
        match self:
            case PromptableBackend.SAM | PromptableBackend.SAM2 | PromptableBackend.SAM3 | PromptableBackend.EDGETAM:
                return frozenset(PromptElement)
            case PromptableBackend.FLORENCE2:
                return frozenset({PromptElement.BOX})

    @property
    def is_mask_only_prompt_supported(self) -> bool:
        """
        Whether a mask alone, without points or a box, can mark an object.

        SAM was trained with mask prompts only next to points, so a mask alone yields no object; SAM 2, SAM 3 and
        EdgeTAM segment from a mask alone.

        Returns
        -------
        bool
            False for ``SAM`` and ``FLORENCE2``, which takes no mask.
        """
        match self:
            case PromptableBackend.SAM | PromptableBackend.FLORENCE2:
                return False
            case PromptableBackend.SAM2 | PromptableBackend.SAM3 | PromptableBackend.EDGETAM:
                return True
