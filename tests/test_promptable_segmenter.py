import numpy as np
import pytest
from geometry import Points2D
from image_container import BinaryImage
from PIL import Image
from stub_promptable import (
    CANDIDATE_SCORES,
    SINGLE_SCORE,
    StubPromptableSegmenter,
    StubSamSegmenter,
    build_promptable_settings,
    patch_checkpoint,
)

from open_vocabulary_segmentation import (
    EncodedImage,
    ImageSize,
    MaskLogits,
    MaskOutput,
    ObjectPrompt,
    ObjectSegmentationResult,
    PromptableBackend,
)
from open_vocabulary_segmentation.array_types import BoolArray

IMAGE_SIZE: ImageSize = ImageSize(width=32, height=32)


def build_image() -> Image.Image:
    return Image.new("RGB", (IMAGE_SIZE.width, IMAGE_SIZE.height))


@pytest.fixture
def segmenter(monkeypatch: pytest.MonkeyPatch) -> StubPromptableSegmenter:
    patch_checkpoint(monkeypatch, "sam2_video")
    return StubPromptableSegmenter(build_promptable_settings(decode_batch_size=2))


def box_mask(x_min: int, y_min: int, x_max: int, y_max: int) -> BoolArray:
    mask: BoolArray = np.zeros((IMAGE_SIZE.height, IMAGE_SIZE.width), dtype=np.bool_)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def test_box_prompts_are_mapped_into_the_model_frame_and_back(segmenter: StubPromptableSegmenter) -> None:
    result: ObjectSegmentationResult = segmenter.segment(
        build_image(), [ObjectPrompt.from_coordinates(box_xyxy=(8, 8, 24, 24))]
    )
    assert np.array_equal(result.masks[0], box_mask(8, 8, 24, 24))
    assert result.scores.tolist() == pytest.approx([SINGLE_SCORE])
    assert result.stability_scores.tolist() == pytest.approx([1.0])
    assert result.pixel_xyxy.tolist() == [[8, 8, 24, 24]]
    assert result[0].box is not None
    assert segmenter.multimask_flags == [False]


def test_auto_output_resolves_single_points_with_the_best_of_three(segmenter: StubPromptableSegmenter) -> None:
    result: ObjectSegmentationResult = segmenter.segment(
        build_image(), [ObjectPrompt.from_coordinates(positive_points=[(16, 16)])]
    )
    assert segmenter.multimask_flags == [True]
    assert len(result) == 1
    assert result.scores.tolist() == pytest.approx([max(CANDIDATE_SCORES)])


def test_all_candidates_keep_three_masks_per_prompt(segmenter: StubPromptableSegmenter) -> None:
    prompts: list[ObjectPrompt] = [
        ObjectPrompt.from_coordinates(box_xyxy=(0, 0, 8, 8)),
        ObjectPrompt.from_coordinates(positive_points=[(16, 16)]),
    ]
    result: ObjectSegmentationResult = segmenter.segment(build_image(), prompts, MaskOutput.ALL_CANDIDATES)
    assert result.prompt_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert result.scores.tolist() == pytest.approx(list(CANDIDATE_SCORES) * 2)
    assert len(result.best_per_prompt()) == 2
    assert result.best_per_prompt().scores.tolist() == pytest.approx([0.9, 0.9])
    assert len(result.of_prompt(1)) == 3


def test_single_output_never_uses_the_three_mask_head(segmenter: StubPromptableSegmenter) -> None:
    segmenter.segment(build_image(), [ObjectPrompt.from_coordinates(positive_points=[(16, 16)])], MaskOutput.SINGLE)
    assert segmenter.multimask_flags == [False]


def test_prompts_are_grouped_by_signature_and_decoded_in_passes(segmenter: StubPromptableSegmenter) -> None:
    prompts: list[ObjectPrompt] = [
        ObjectPrompt.from_coordinates(box_xyxy=(0, 0, 8, 8)),
        ObjectPrompt.from_coordinates(positive_points=[(4, 4), (6, 6)]),
        ObjectPrompt.from_coordinates(box_xyxy=(8, 8, 16, 16)),
        ObjectPrompt.from_coordinates(box_xyxy=(16, 16, 24, 24)),
        ObjectPrompt(mask=BinaryImage(value=box_mask(0, 0, 16, 16))),
        ObjectPrompt(mask=BinaryImage(value=box_mask(16, 16, 32, 32))),
    ]
    result: ObjectSegmentationResult = segmenter.segment(build_image(), prompts)
    assert segmenter.decoded_prompt_counts == [2, 1, 1, 1, 1]
    assert result.prompt_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert result.pixel_xyxy[2].tolist() == [8, 8, 16, 16]
    assert np.array_equal(result.masks[4], box_mask(0, 0, 16, 16))


def test_negative_points_do_not_paint(segmenter: StubPromptableSegmenter) -> None:
    prompt: ObjectPrompt = ObjectPrompt.from_coordinates(positive_points=[(4, 4)], negative_points=[(28, 28)])
    result: ObjectSegmentationResult = segmenter.segment(build_image(), [prompt])
    assert bool(result.masks[0][4, 4])
    assert not bool(result.masks[0][28, 28])


def test_encodings_are_reused_and_checked(segmenter: StubPromptableSegmenter, monkeypatch: pytest.MonkeyPatch) -> None:
    encoding: EncodedImage = segmenter.encode(build_image())
    assert encoding.image_size == IMAGE_SIZE
    segmenter.segment(encoding, [ObjectPrompt.from_coordinates(box_xyxy=(0, 0, 8, 8))])
    segmenter.segment(encoding, [ObjectPrompt.from_coordinates(box_xyxy=(8, 8, 16, 16))])
    assert segmenter.embedded_batch_sizes == [1]
    other: StubPromptableSegmenter = StubPromptableSegmenter(build_promptable_settings(weights_path="other"))
    with pytest.raises(ValueError, match="encoded by"):
        other.segment(encoding, [ObjectPrompt.from_coordinates(box_xyxy=(0, 0, 8, 8))])


def test_raw_images_are_encoded_per_mini_batch(segmenter: StubPromptableSegmenter) -> None:
    encoding: EncodedImage = segmenter.encode(build_image())
    prompts: list[ObjectPrompt] = [ObjectPrompt.from_coordinates(box_xyxy=(0, 0, 8, 8))]
    results: list[ObjectSegmentationResult] = segmenter.segment_images(
        [build_image(), encoding, build_image(), build_image()], [prompts] * 4
    )
    assert len(results) == 4
    assert segmenter.embedded_batch_sizes == [1, 1, 2]


def test_refinement_feeds_back_the_mask_logits(segmenter: StubPromptableSegmenter) -> None:
    first: ObjectSegmentationResult = segmenter.segment(
        build_image(), [ObjectPrompt.from_coordinates(box_xyxy=(8, 8, 24, 24))]
    )
    refined_prompt: ObjectPrompt = first[0].refine(negative_points=Points2D(value=np.array([[2.0, 2.0]])))
    assert isinstance(refined_prompt.mask, MaskLogits)
    assert refined_prompt.point_count == 1
    refined: ObjectSegmentationResult = segmenter.segment(build_image(), [refined_prompt])
    assert np.array_equal(refined.masks[0], box_mask(8, 8, 24, 24))


def test_invalid_prompts_are_rejected_before_inference(segmenter: StubPromptableSegmenter) -> None:
    with pytest.raises(ValueError, match="inside the 32x32 image"):
        segmenter.segment(build_image(), [ObjectPrompt.from_coordinates(positive_points=[(40, 4)])])
    with pytest.raises(ValueError, match="mask must have shape"):
        segmenter.segment(build_image(), [ObjectPrompt(mask=BinaryImage(value=np.zeros((4, 4), dtype=np.bool_)))])
    foreign_logits: MaskLogits = MaskLogits(
        logits=np.zeros((16, 16), dtype=np.float32),
        backend=PromptableBackend.SAM,
        weights_path="stub",
        image_size=IMAGE_SIZE,
    )
    with pytest.raises(ValueError, match="mask logits must come from this segmenter"):
        segmenter.segment(build_image(), [ObjectPrompt(mask=foreign_logits)])
    wrong_resolution: MaskLogits = MaskLogits(
        logits=np.zeros((8, 8), dtype=np.float32),
        backend=PromptableBackend.SAM2,
        weights_path="stub",
        image_size=IMAGE_SIZE,
    )
    with pytest.raises(ValueError, match="mask logits must have shape"):
        segmenter.segment(build_image(), [ObjectPrompt(mask=wrong_resolution)])
    with pytest.raises(ValueError, match="expected prompts for 1 images"):
        segmenter.segment_images([build_image()], [])
    assert segmenter.decoded_prompt_counts == []


def test_sam_rejects_masks_without_points_or_box(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_checkpoint(monkeypatch, "sam")
    sam: StubSamSegmenter = StubSamSegmenter(build_promptable_settings(PromptableBackend.SAM))
    with pytest.raises(ValueError, match="cannot segment from a mask alone"):
        sam.segment(build_image(), [ObjectPrompt(mask=BinaryImage(value=box_mask(0, 0, 8, 8)))])
    result: ObjectSegmentationResult = sam.segment(
        build_image(),
        [ObjectPrompt.from_coordinates(positive_points=[(4, 4)], mask=BinaryImage(value=box_mask(0, 0, 8, 8)))],
    )
    assert len(result) == 1


def test_empty_prompt_list_returns_an_empty_result(segmenter: StubPromptableSegmenter) -> None:
    result: ObjectSegmentationResult = segmenter.segment(build_image(), [])
    assert len(result) == 0
    assert result.image_size == IMAGE_SIZE


def test_settings_and_checkpoint_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_checkpoint(monkeypatch, "sam")
    with pytest.raises(ValueError, match="is a 'sam' checkpoint"):
        StubPromptableSegmenter(build_promptable_settings())
    with pytest.raises(ValueError, match="needs sam2 settings"):
        StubPromptableSegmenter(build_promptable_settings(PromptableBackend.SAM))
    with pytest.raises(ValueError, match="decode_batch_size"):
        build_promptable_settings(decode_batch_size=0)
