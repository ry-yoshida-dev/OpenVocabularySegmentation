from enum import StrEnum


class MaskOutput(StrEnum):
    """
    Which of the masks a promptable decoder predicts are returned for each object prompt.

    SAM-family decoders have a single-mask head and a three-mask head; the three masks resolve ambiguous prompts
    (whole, part, sub-part) and come with predicted IoU scores.

    Attributes
    ----------
    AUTO : str
        The best of three masks for ambiguous prompts (one point, no box, no mask), the single-mask head otherwise;
        the behavior recommended by SAM.
    SINGLE : str
        Always the single-mask head.
    BEST_OF_MULTIPLE : str
        Always the three-mask head, keeping the mask with the highest predicted IoU.
    ALL_CANDIDATES : str
        Always the three-mask head, keeping every mask; each object prompt yields three results.
    """

    AUTO = "auto"
    SINGLE = "single"
    BEST_OF_MULTIPLE = "best_of_multiple"
    ALL_CANDIDATES = "all_candidates"

    def is_multimask(self, is_ambiguous: bool) -> bool:
        """
        Whether the three-mask head is used for a prompt.

        Parameters
        ----------
        is_ambiguous : bool
            Whether the prompt is a single point without box or mask.

        Returns
        -------
        bool
            True if the decoder must predict three masks.
        """
        match self:
            case MaskOutput.AUTO:
                return is_ambiguous
            case MaskOutput.SINGLE:
                return False
            case MaskOutput.BEST_OF_MULTIPLE | MaskOutput.ALL_CANDIDATES:
                return True

    @property
    def is_every_candidate_kept(self) -> bool:
        """
        Whether every predicted mask is returned instead of the best one.

        Returns
        -------
        bool
            True for ``ALL_CANDIDATES``.
        """
        match self:
            case MaskOutput.ALL_CANDIDATES:
                return True
            case MaskOutput.AUTO | MaskOutput.SINGLE | MaskOutput.BEST_OF_MULTIPLE:
                return False
