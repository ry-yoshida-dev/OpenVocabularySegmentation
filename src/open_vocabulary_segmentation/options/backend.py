from enum import StrEnum

from .promptable_backend import PromptableBackend


class SegmenterBackend(StrEnum):
    """
    Segmenter family.

    Model-specific values (weights, recommended thresholds) live in the YAML presets; only what code must know about
    a family, whether it needs boxes from an open-vocabulary detector, is defined here.

    Attributes
    ----------
    SAM : str
        Segment Anything through Hugging Face ``transformers``; masks the boxes of a detector.
    SAM2 : str
        SAM 2 and SAM 2.1 through Hugging Face ``transformers``; masks the boxes of a detector.
    SAM3 : str
        SAM 3 through Hugging Face ``transformers``; segments every instance of a text phrase on its own.
    EDGETAM : str
        EdgeTAM through Hugging Face ``transformers``; masks the boxes of a detector.
    FLORENCE2 : str
        Florence-2 through Hugging Face ``transformers``; detects every instance of a text phrase, then writes the
        polygon of each.
    FLORENCE2_REFERRING : str
        Florence-2 referring expression segmentation; writes the polygons of the one object each phrase refers to.
    """

    SAM = "sam"
    SAM2 = "sam2"
    SAM3 = "sam3"
    EDGETAM = "edgetam"
    FLORENCE2 = "florence2"
    FLORENCE2_REFERRING = "florence2_referring"

    @property
    def promptable_backend(self) -> PromptableBackend | None:
        """
        Promptable model family masking the detector boxes of a box-prompted family.

        Returns
        -------
        PromptableBackend | None
            The matching promptable family for ``SAM``, ``SAM2`` and ``EDGETAM``; ``None`` for families finding
            instances themselves.
        """
        match self:
            case SegmenterBackend.SAM:
                return PromptableBackend.SAM
            case SegmenterBackend.SAM2:
                return PromptableBackend.SAM2
            case SegmenterBackend.EDGETAM:
                return PromptableBackend.EDGETAM
            case SegmenterBackend.SAM3 | SegmenterBackend.FLORENCE2 | SegmenterBackend.FLORENCE2_REFERRING:
                return None

    @property
    def is_box_prompted(self) -> bool:
        """
        Whether the family only masks given boxes and so needs an open-vocabulary detector to find them.

        Returns
        -------
        bool
            True for ``SAM``, ``SAM2`` and ``EDGETAM``.
        """
        match self:
            case SegmenterBackend.SAM | SegmenterBackend.SAM2 | SegmenterBackend.EDGETAM:
                return True
            case SegmenterBackend.SAM3 | SegmenterBackend.FLORENCE2 | SegmenterBackend.FLORENCE2_REFERRING:
                return False
