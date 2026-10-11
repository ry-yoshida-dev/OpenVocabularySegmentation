from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np

from ..array_types import BoolArray, IntArray
from .region_kind import RegionKind


@dataclass(frozen=True)
class RegionCleaner:
    """
    Removes small holes and islands from binary masks, as SAM's ``remove_small_regions``.

    Regions are 8-connected. When every island of a mask is small, the largest one is kept so the mask never becomes
    empty.

    Attributes
    ----------
    minimum_area : int
        Holes and islands with fewer pixels are removed.

    Raises
    ------
    ValueError
        If ``minimum_area`` is negative.
    """

    CONNECTIVITY: ClassVar[int] = 8

    minimum_area: int

    def __post_init__(self) -> None:
        if self.minimum_area < 0:
            raise ValueError(f"minimum_area must not be negative. got {self.minimum_area}")

    def clean(self, mask: BoolArray) -> BoolArray:
        """
        Fill small holes, then erase small islands.

        Parameters
        ----------
        mask : BoolArray
            Binary mask, shape (H, W).

        Returns
        -------
        BoolArray
            Cleaned mask, shape (H, W); ``mask`` itself when nothing changes.
        """
        return self.remove(self.remove(mask, RegionKind.HOLE), RegionKind.ISLAND)

    def remove(self, mask: BoolArray, region_kind: RegionKind) -> BoolArray:
        """
        Remove the small regions of one kind.

        Parameters
        ----------
        mask : BoolArray
            Binary mask, shape (H, W).
        region_kind : RegionKind
            Holes to fill or islands to erase.

        Returns
        -------
        BoolArray
            Mask without the small regions, shape (H, W); ``mask`` itself when nothing changes.
        """
        is_hole: bool = region_kind is RegionKind.HOLE
        working_mask: np.ndarray = (mask ^ is_hole).astype(np.uint8)
        label_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
            working_mask, connectivity=self.CONNECTIVITY
        )
        region_areas: IntArray = statistics[1:, cv2.CC_STAT_AREA].astype(np.int64)
        small_labels: list[int] = [int(label) + 1 for label in np.flatnonzero(region_areas < self.minimum_area)]
        if not small_labels:
            return mask
        match region_kind:
            case RegionKind.HOLE:
                kept_labels: list[int] = [0, *small_labels]
            case RegionKind.ISLAND:
                small_label_set: set[int] = set(small_labels)
                kept_labels = [label for label in range(1, label_count) if label not in small_label_set]
                if not kept_labels:
                    kept_labels = [int(np.argmax(region_areas)) + 1]
        return np.isin(labels, kept_labels)
