from enum import StrEnum


class PromptElement(StrEnum):
    """
    Kind of input telling a promptable model which object to segment.

    Attributes
    ----------
    POINT : str
        Positive points on the object or negative points off it.
    BOX : str
        Box around the object.
    MASK : str
        Rough mask of the object, either a binary mask or the mask logits of a previous prediction.
    """

    POINT = "point"
    BOX = "box"
    MASK = "mask"
