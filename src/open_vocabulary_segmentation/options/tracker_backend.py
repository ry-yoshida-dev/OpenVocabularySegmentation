from enum import StrEnum


class TrackerBackend(StrEnum):
    """
    Model family that tracks prompted objects through a video with a memory of past frames.

    Attributes
    ----------
    SAM2 : str
        SAM 2 and SAM 2.1 video predictor through Hugging Face ``transformers`` (``Sam2VideoModel``).
    SAM3 : str
        SAM 3 tracker through Hugging Face ``transformers`` (``Sam3TrackerVideoModel``).
    EDGETAM : str
        EdgeTAM video predictor through Hugging Face ``transformers`` (``EdgeTamVideoModel``): SAM 2 with a light
        image encoder and a spatial perceiver over its memory; needs the ``timm`` package.
    """

    SAM2 = "sam2"
    SAM3 = "sam3"
    EDGETAM = "edgetam"

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
            case TrackerBackend.SAM2:
                return frozenset({"sam2_video"})
            case TrackerBackend.SAM3:
                return frozenset({"sam3_video", "sam3_tracker_video"})
            case TrackerBackend.EDGETAM:
                return frozenset({"edgetam_video"})
