from dataclasses import dataclass, field

from ..array_types import BoolArray, FloatArray, IntArray


@dataclass
class CropCandidates:
    """
    Masks of one crop kept by the score filters, collected batch by batch before NMS.

    Attributes
    ----------
    masks : list[BoolArray]
        Masks in crop pixels, shape (h, w) each.
    scores : list[float]
        Predicted IoU of each mask.
    stability_scores : list[float]
        Stability score of each mask.
    point_xy : list[FloatArray]
        Prompting grid point of each mask in image pixels, shape (2,) each.
    pixel_xyxy : list[IntArray]
        Bounding box of each mask in crop pixels, shape (4,) each.
    """

    masks: list[BoolArray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    stability_scores: list[float] = field(default_factory=list)
    point_xy: list[FloatArray] = field(default_factory=list)
    pixel_xyxy: list[IntArray] = field(default_factory=list)

    def select(self, indices: IntArray) -> "CropCandidates":
        """
        Keep a subset of candidates.

        Parameters
        ----------
        indices : IntArray
            Kept indices, in the order to keep.

        Returns
        -------
        CropCandidates
            Candidates at ``indices``.
        """
        positions: list[int] = [int(index) for index in indices]
        return CropCandidates(
            masks=[self.masks[position] for position in positions],
            scores=[self.scores[position] for position in positions],
            stability_scores=[self.stability_scores[position] for position in positions],
            point_xy=[self.point_xy[position] for position in positions],
            pixel_xyxy=[self.pixel_xyxy[position] for position in positions],
        )
