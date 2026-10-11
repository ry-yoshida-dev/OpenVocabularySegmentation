from enum import IntEnum


class PointLabel(IntEnum):
    """
    Point labels read by the prompt encoders of SAM-family models.

    Attributes
    ----------
    POSITIVE : int
        Point on the object.
    NEGATIVE : int
        Point off the object.
    NOT_A_POINT : int
        Placeholder standing for no point, used when a prompt has neither points nor a box.
    """

    POSITIVE = 1
    NEGATIVE = 0
    NOT_A_POINT = -1
