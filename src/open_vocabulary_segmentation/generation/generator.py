from collections.abc import Sequence

import numpy as np
import torch
from open_vocabulary_detector import ImageSize
from PIL import Image
from torchvision.ops import batched_nms

from ..array_types import BoolArray, FloatArray, IntArray
from ..image import ImageInput, RgbImage
from ..object_prompt import ObjectPrompt
from ..options import PromptElement
from ..promptable import EncodedImage, PromptableSegmenter
from ..result import MaskBounds, MaskGenerationResult, ObjectSegmentationResult
from .crop_box import CropBox
from .crop_candidates import CropCandidates
from .options import MaskGenerationOptions
from .region_cleaner import RegionCleaner


class AutomaticMaskGenerator:
    """
    Segments everything in an image with a promptable segmenter, following SAM's ``SamAutomaticMaskGenerator``.

    Works with every SAM-family promptable backend (SAM, SAM 2, SAM 3, EdgeTAM). For every crop of
    ``CropBox.layout`` the crop is encoded once and prompted with a grid of single points in batches of
    ``points_per_batch``. Masks below the predicted IoU or stability thresholds, empty masks and masks cut by an inner
    crop edge are dropped, the rest deduplicated by box NMS within the crop; kept masks are copied out of their batch
    so it is freed. With several crops, the boxes of all crops are deduplicated by NMS preferring masks of smaller
    crops before the survivors are placed in full-image masks. With ``minimum_region_area``, small holes and islands
    are removed and the masks changed by it lose ties in a final NMS.

    Attributes
    ----------
    segmenter : PromptableSegmenter
        Model prompted with the point grids.
    options : MaskGenerationOptions
        Grid, crop and filtering parameters.
    """

    def __init__(self, segmenter: PromptableSegmenter, options: MaskGenerationOptions) -> None:
        """
        Parameters
        ----------
        segmenter : PromptableSegmenter
            Model prompted with the point grids.
        options : MaskGenerationOptions
            Grid, crop and filtering parameters.

        Raises
        ------
        ValueError
            If the segmenter does not accept point prompts.
        """
        if PromptElement.POINT not in segmenter.supported_prompt_elements:
            raise ValueError(f"mask generation needs point prompts, which {type(segmenter).__name__} does not accept.")
        self.segmenter: PromptableSegmenter = segmenter
        self.options: MaskGenerationOptions = options

    def generate(self, image: ImageInput) -> MaskGenerationResult:
        """
        Find every region of an image.

        Parameters
        ----------
        image : ImageInput
            PIL image of any mode or ``image_container`` container of any channel order.

        Returns
        -------
        MaskGenerationResult
            Regions, unordered.
        """
        rgb_image: Image.Image = RgbImage.from_input(image)
        image_size: ImageSize = ImageSize.from_image(rgb_image)
        crops: list[CropBox] = CropBox.layout(
            image_size, self.options.crop_layer_count, self.options.crop_overlap_ratio
        )
        crop_candidates: list[tuple[CropBox, CropCandidates]] = [
            (crop, self._segment_crop(rgb_image, crop, image_size)) for crop in crops
        ]
        image_xyxy: list[IntArray] = []
        crop_preferences: list[float] = []
        for crop, candidates in crop_candidates:
            offset_xyxy: IntArray = np.array([crop.x_min, crop.y_min, crop.x_min, crop.y_min], dtype=np.int64)
            image_xyxy.extend(xyxy + offset_xyxy for xyxy in candidates.pixel_xyxy)
            crop_preferences.extend(1.0 / crop.area for _ in candidates.masks)
        kept_positions: set[int] = set(range(len(image_xyxy)))
        if len(crops) > 1 and image_xyxy:
            kept_positions = set(
                self._box_nms(
                    np.stack(image_xyxy),
                    np.array(crop_preferences, dtype=np.float64),
                    self.options.crop_nms_threshold,
                ).tolist()
            )
        masks: list[BoolArray] = []
        scores: list[float] = []
        stability_scores: list[float] = []
        point_xy: list[FloatArray] = []
        crop_xyxy: list[IntArray] = []
        position: int = 0
        for crop, candidates in crop_candidates:
            for mask, score, stability_score, point in zip(
                candidates.masks, candidates.scores, candidates.stability_scores, candidates.point_xy, strict=True
            ):
                if position in kept_positions:
                    full_mask: BoolArray = np.zeros((image_size.height, image_size.width), dtype=np.bool_)
                    full_mask[crop.y_min : crop.y_max, crop.x_min : crop.x_max] = mask
                    masks.append(full_mask)
                    scores.append(score)
                    stability_scores.append(stability_score)
                    point_xy.append(point)
                    crop_xyxy.append(crop.xyxy)
                position += 1
        result: MaskGenerationResult = MaskGenerationResult(
            masks=np.stack(masks) if masks else np.zeros((0, image_size.height, image_size.width), dtype=np.bool_),
            scores=np.array(scores, dtype=np.float64),
            stability_scores=np.array(stability_scores, dtype=np.float64),
            point_xy=np.array(point_xy, dtype=np.float64).reshape(-1, 2),
            crop_xyxy=np.array(crop_xyxy, dtype=np.float64).reshape(-1, 4),
            image_size=image_size,
        )
        if self.options.minimum_region_area > 0 and len(result):
            result = self._remove_small_regions(result)
        return result

    def generate_images(self, images: Sequence[ImageInput]) -> list[MaskGenerationResult]:
        """
        Find every region of many images.

        Parameters
        ----------
        images : Sequence[ImageInput]
            Input images.

        Returns
        -------
        list[MaskGenerationResult]
            One result per image, in input order.
        """
        return [self.generate(image) for image in images]

    def _segment_crop(self, image: Image.Image, crop: CropBox, image_size: ImageSize) -> CropCandidates:
        crop_image: Image.Image = image.crop((crop.x_min, crop.y_min, crop.x_max, crop.y_max))
        encoding: EncodedImage = self.segmenter.encode(crop_image)
        grid: FloatArray = crop.point_grid(self.options.points_per_side_of(crop.layer_index))
        offset_xy: FloatArray = np.array([crop.x_min, crop.y_min], dtype=np.float64)
        candidates: CropCandidates = CropCandidates()
        for start in range(0, len(grid), self.options.points_per_batch):
            batch: FloatArray = grid[start : start + self.options.points_per_batch]
            prompts: list[ObjectPrompt] = [
                ObjectPrompt.from_coordinates(positive_points=[(float(x), float(y))]) for x, y in batch
            ]
            result: ObjectSegmentationResult = self.segmenter.segment(encoding, prompts, self.options.mask_output)
            image_xyxy: IntArray = result.pixel_xyxy + np.tile(offset_xy.astype(np.int64), 2)
            is_kept: BoolArray = (
                (result.scores >= self.options.predicted_iou_threshold)
                & (result.stability_scores >= self.options.stability_score_threshold)
                & ~result.is_empty_mask
                & ~crop.is_cutting(image_xyxy, image_size)
            )
            for index in np.flatnonzero(is_kept).tolist():
                candidates.masks.append(result.masks[index].copy())
                candidates.scores.append(float(result.scores[index]))
                candidates.stability_scores.append(float(result.stability_scores[index]))
                candidates.point_xy.append(batch[int(result.prompt_indices[index])] + offset_xy)
                candidates.pixel_xyxy.append(result.pixel_xyxy[index])
        if not candidates.masks:
            return candidates
        kept_indices: IntArray = self._box_nms(
            np.stack(candidates.pixel_xyxy),
            np.array(candidates.scores, dtype=np.float64),
            self.options.box_nms_threshold,
        )
        return candidates.select(kept_indices)

    def _remove_small_regions(self, result: MaskGenerationResult) -> MaskGenerationResult:
        cleaner: RegionCleaner = RegionCleaner(self.options.minimum_region_area)
        cleaned_masks: list[BoolArray] = []
        is_unchanged: list[bool] = []
        for mask in result.masks:
            cleaned: BoolArray = cleaner.clean(mask)
            cleaned_masks.append(cleaned)
            is_unchanged.append(cleaned is mask)
        masks: BoolArray = np.stack(cleaned_masks)
        is_filled: BoolArray = masks.any(axis=(1, 2))
        cleaned_result: MaskGenerationResult = MaskGenerationResult(
            masks=masks[is_filled],
            scores=result.scores[is_filled],
            stability_scores=result.stability_scores[is_filled],
            point_xy=result.point_xy[is_filled],
            crop_xyxy=result.crop_xyxy[is_filled],
            image_size=result.image_size,
        )
        preference: FloatArray = np.array(is_unchanged, dtype=np.float64)[is_filled]
        return cleaned_result.select(
            self._box_nms(MaskBounds.pixel_xyxy(cleaned_result.masks), preference, self.options.box_nms_threshold)
        )

    @staticmethod
    def _box_nms(pixel_xyxy: IntArray, scores: FloatArray, iou_threshold: float) -> IntArray:
        kept: torch.Tensor = batched_nms(
            torch.from_numpy(pixel_xyxy.astype(np.float32)).reshape(-1, 4),
            torch.from_numpy(scores.astype(np.float32)),
            torch.zeros(len(scores), dtype=torch.int64),
            iou_threshold,
        )
        return kept.numpy().astype(np.int64)
