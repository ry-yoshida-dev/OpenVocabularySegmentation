import numpy as np
import pytest
from PIL import Image
from stub_promptable import StubPromptableSegmenter, build_promptable_settings, patch_checkpoint

from open_vocabulary_segmentation import (
    AutomaticMaskGenerator,
    ImageSize,
    MaskGenerationOptions,
    MaskGenerationResult,
    MaskGeneratorSettings,
    PromptableBackend,
)
from open_vocabulary_segmentation.array_types import BoolArray
from open_vocabulary_segmentation.generation import CropBox, RegionCleaner, RegionKind

IMAGE_SIZE: ImageSize = ImageSize(width=64, height=48)


@pytest.fixture
def segmenter(monkeypatch: pytest.MonkeyPatch) -> StubPromptableSegmenter:
    patch_checkpoint(monkeypatch, "sam2_video")
    return StubPromptableSegmenter(build_promptable_settings())


def test_crop_layout_covers_the_image_with_overlapping_crops() -> None:
    crops: list[CropBox] = CropBox.layout(IMAGE_SIZE, layer_count=2, overlap_ratio=512 / 1500)
    assert len(crops) == 1 + 4 + 16
    assert crops[0].xyxy.tolist() == [0, 0, 64, 48]
    covered: BoolArray = np.zeros((48, 64), dtype=np.bool_)
    for crop in crops[1:5]:
        assert crop.layer_index == 1
        covered[crop.y_min : crop.y_max, crop.x_min : crop.x_max] = True
    assert bool(covered.all())
    assert crops[1].x_max > crops[3].x_min


def test_point_grid_and_cut_detection() -> None:
    crop: CropBox = CropBox(x_min=0, y_min=0, x_max=32, y_max=24, layer_index=1)
    grid = crop.point_grid(2)
    assert grid.tolist() == [[8, 6], [24, 6], [8, 18], [24, 18]]
    large_crop: CropBox = CropBox(x_min=0, y_min=0, x_max=60, y_max=60, layer_index=1)
    is_cut = large_crop.is_cutting(np.array([[0, 0, 59, 10], [0, 0, 5, 5]]), ImageSize(width=120, height=120))
    assert is_cut.tolist() == [True, False]
    whole: CropBox = CropBox.layout(IMAGE_SIZE, 0, 0.0)[0]
    assert not bool(whole.is_cutting(np.array([[0, 0, 64, 48]]), IMAGE_SIZE).any())
    with pytest.raises(ValueError, match="invalid crop"):
        CropBox(x_min=5, y_min=0, x_max=5, y_max=4, layer_index=0)


def test_region_cleaner_fills_holes_and_erases_islands() -> None:
    mask: BoolArray = np.zeros((20, 20), dtype=np.bool_)
    mask[2:15, 2:15] = True
    mask[5, 5] = False
    mask[18, 18] = True
    cleaner: RegionCleaner = RegionCleaner(minimum_area=5)
    cleaned: BoolArray = cleaner.clean(mask)
    assert bool(cleaned[5, 5]) and not bool(cleaned[18, 18])
    assert cleaner.remove(cleaned, RegionKind.HOLE) is cleaned
    lonely: BoolArray = np.zeros((10, 10), dtype=np.bool_)
    lonely[1, 1] = True
    assert bool(RegionCleaner(minimum_area=5).remove(lonely, RegionKind.ISLAND)[1, 1])


def test_generation_options_validate() -> None:
    assert MaskGenerationOptions(points_per_side=32, crop_points_downscale_factor=2).points_per_side_of(2) == 8
    with pytest.raises(ValueError, match="points_per_side"):
        MaskGenerationOptions(points_per_side=0)
    with pytest.raises(ValueError, match="predicted_iou_threshold"):
        MaskGenerationOptions(predicted_iou_threshold=1.5)
    with pytest.raises(ValueError, match="crop_overlap_ratio"):
        MaskGenerationOptions(crop_overlap_ratio=1.0)


def test_generator_prompts_a_grid_and_deduplicates(segmenter: StubPromptableSegmenter) -> None:
    options: MaskGenerationOptions = MaskGenerationOptions(
        points_per_side=4, points_per_batch=5, predicted_iou_threshold=0.85, stability_score_threshold=0.9
    )
    result: MaskGenerationResult = AutomaticMaskGenerator(segmenter, options).generate(
        Image.new("RGB", (IMAGE_SIZE.width, IMAGE_SIZE.height))
    )
    assert segmenter.decoded_prompt_counts == [5, 5, 5, 1]
    assert len(result) == 16
    assert bool((result.scores >= 0.85).all())
    assert result.point_xy.shape == (16, 2)
    assert result.crop_xyxy[0].tolist() == [0, 0, 64, 48]


def test_generator_with_crops_and_region_cleanup(segmenter: StubPromptableSegmenter) -> None:
    options: MaskGenerationOptions = MaskGenerationOptions(
        points_per_side=2,
        predicted_iou_threshold=0.0,
        stability_score_threshold=0.0,
        crop_layer_count=1,
        minimum_region_area=4,
    )
    result: MaskGenerationResult = AutomaticMaskGenerator(segmenter, options).generate(
        Image.new("RGB", (IMAGE_SIZE.width, IMAGE_SIZE.height))
    )
    assert len(result) >= 1
    assert segmenter.embedded_batch_sizes == [1, 1, 1, 1, 1]


def test_generation_result_tools() -> None:
    masks: BoolArray = np.zeros((2, 10, 10), dtype=np.bool_)
    masks[0, :8, :8] = True
    masks[1, 2:4, 2:4] = True
    result: MaskGenerationResult = MaskGenerationResult(
        masks=masks,
        scores=np.array([0.9, 0.95]),
        stability_scores=np.array([0.97, 0.99]),
        point_xy=np.array([[1.0, 1.0], [3.0, 3.0]]),
        crop_xyxy=np.array([[0.0, 0.0, 10.0, 10.0]] * 2),
        image_size=ImageSize(width=10, height=10),
    )
    assert result.sort_by_area().areas.tolist() == [64, 4]
    assert result.sort_by_area(is_descending=False).areas.tolist() == [4, 64]
    assert result.sort_by_score().scores.tolist() == pytest.approx([0.95, 0.9])
    assert len(result.filter_by_area(10)) == 1
    assert int(result.instance_map()[3, 3]) == 1
    assert result[0].box.value.tolist() == [0, 0, 8, 8]
    assert result.boxes.value.shape == (2, 4)
    with pytest.raises(ValueError, match="must not be empty"):
        MaskGenerationResult(
            masks=np.zeros((1, 10, 10), dtype=np.bool_),
            scores=np.array([0.9]),
            stability_scores=np.array([0.9]),
            point_xy=np.zeros((1, 2)),
            crop_xyxy=np.zeros((1, 4)),
            image_size=ImageSize(width=10, height=10),
        )


def test_generator_rejects_backends_without_point_prompts() -> None:
    settings: MaskGeneratorSettings = MaskGeneratorSettings(
        segmenter=build_promptable_settings(backend=PromptableBackend.FLORENCE2)
    )
    with pytest.raises(ValueError, match="point prompts"):
        settings.build()
