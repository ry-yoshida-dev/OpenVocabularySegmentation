from enum import StrEnum


class PropagationDirection(StrEnum):
    """
    Direction in which a video tracker propagates its prompted objects from a start frame.

    Attributes
    ----------
    FORWARD : str
        Towards later frames.
    BACKWARD : str
        Towards earlier frames.
    BIDIRECTIONAL : str
        Forward to the end, then backward to the start.
    """

    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"

    @property
    def is_reverse_per_pass(self) -> tuple[bool, ...]:
        """
        Passes run for the direction, each flagged by whether it moves towards earlier frames.

        Returns
        -------
        tuple[bool, ...]
            ``(False,)``, ``(True,)`` or ``(False, True)``.
        """
        match self:
            case PropagationDirection.FORWARD:
                return (False,)
            case PropagationDirection.BACKWARD:
                return (True,)
            case PropagationDirection.BIDIRECTIONAL:
                return (False, True)
