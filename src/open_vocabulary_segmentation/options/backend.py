from enum import StrEnum


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
    FLORENCE2 : str
        Florence-2 through Hugging Face ``transformers``; detects every instance of a text phrase, then writes the
        polygon of each.
    """

    SAM = "sam"
    SAM2 = "sam2"
    SAM3 = "sam3"
    FLORENCE2 = "florence2"

    @property
    def is_box_prompted(self) -> bool:
        """
        Whether the family only masks given boxes and so needs an open-vocabulary detector to find them.

        Returns
        -------
        bool
            True for ``SAM`` and ``SAM2``.
        """
        match self:
            case SegmenterBackend.SAM | SegmenterBackend.SAM2:
                return True
            case SegmenterBackend.SAM3 | SegmenterBackend.FLORENCE2:
                return False
