from dataclasses import dataclass

from ..options import MaskOutput


@dataclass(frozen=True, kw_only=True)
class MaskGenerationOptions:
    """
    Parameters of automatic mask generation ("segment everything"), named after those of SAM's
    ``SamAutomaticMaskGenerator``.

    A grid of single-point prompts covers the image and, with ``crop_layer_count > 0``, overlapping crops of it at
    finer scales; masks are kept by predicted IoU and stability, deduplicated by box NMS within a crop and across
    crops, and optionally cleaned of small holes and islands.

    Attributes
    ----------
    points_per_side : int
        Grid points along each side of the image (and of every crop of the first layer).
    points_per_batch : int
        Point prompts decoded per pass; higher is faster and uses more memory.
    mask_output : MaskOutput
        Masks kept per point; SAM keeps all three candidates.
    predicted_iou_threshold : float
        Minimum predicted IoU of a kept mask.
    stability_score_threshold : float
        Minimum stability score of a kept mask.
    box_nms_threshold : float
        Box IoU above which the less confident of two masks of one crop is removed.
    crop_layer_count : int
        Number of crop layers; layer ``i`` splits the image into ``2^i x 2^i`` overlapping crops.
    crop_nms_threshold : float
        Box IoU above which a mask of a larger crop is removed in favor of one of a smaller crop.
    crop_overlap_ratio : float
        Overlap of neighboring crops relative to the crop size of the first layer.
    crop_points_downscale_factor : int
        Points per side are divided by this factor at every further crop layer.
    minimum_region_area : int
        Holes and disconnected islands smaller than this many pixels are removed from each mask; ``0`` keeps masks
        as predicted.

    Raises
    ------
    ValueError
        If a count is not positive, a threshold is outside ``[0, 1]`` or the overlap ratio is outside ``[0, 1)``.
    """

    points_per_side: int = 32
    points_per_batch: int = 64
    mask_output: MaskOutput = MaskOutput.ALL_CANDIDATES
    predicted_iou_threshold: float = 0.88
    stability_score_threshold: float = 0.95
    box_nms_threshold: float = 0.7
    crop_layer_count: int = 0
    crop_nms_threshold: float = 0.7
    crop_overlap_ratio: float = 512 / 1500
    crop_points_downscale_factor: int = 1
    minimum_region_area: int = 0

    def __post_init__(self) -> None:
        for name, count in (
            ("points_per_side", self.points_per_side),
            ("points_per_batch", self.points_per_batch),
            ("crop_points_downscale_factor", self.crop_points_downscale_factor),
        ):
            if count <= 0:
                raise ValueError(f"{name} must be positive. got {count}")
        for name, count in (
            ("crop_layer_count", self.crop_layer_count),
            ("minimum_region_area", self.minimum_region_area),
        ):
            if count < 0:
                raise ValueError(f"{name} must not be negative. got {count}")
        for name, threshold in (
            ("predicted_iou_threshold", self.predicted_iou_threshold),
            ("stability_score_threshold", self.stability_score_threshold),
            ("box_nms_threshold", self.box_nms_threshold),
            ("crop_nms_threshold", self.crop_nms_threshold),
        ):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]. got {threshold}")
        if not 0.0 <= self.crop_overlap_ratio < 1.0:
            raise ValueError(f"crop_overlap_ratio must be in [0, 1). got {self.crop_overlap_ratio}")

    def points_per_side_of(self, layer_index: int) -> int:
        """
        Grid points along each side of the crops of one layer.

        Parameters
        ----------
        layer_index : int
            Crop layer; ``0`` is the whole image.

        Returns
        -------
        int
            At least one point per side.
        """
        divisor: int = self.crop_points_downscale_factor**layer_index
        return max(1, self.points_per_side // divisor)
