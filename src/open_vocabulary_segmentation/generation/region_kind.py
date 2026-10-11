from enum import StrEnum


class RegionKind(StrEnum):
    """
    Kind of small connected region removed from a mask.

    Attributes
    ----------
    HOLE : str
        Background region enclosed by the mask, filled.
    ISLAND : str
        Mask region disconnected from the rest of the mask, erased.
    """

    HOLE = "hole"
    ISLAND = "island"
