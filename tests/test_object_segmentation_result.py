import numpy as np
import pytest
from geometry import Points2D

from open_vocabulary_segmentation import (
    ImageSize,
    MaskLogits,
    ObjectPrompt,
    ObjectSegmentation,
    ObjectSegmentationResult,
    Prompt,
    PromptableBackend,
    SemanticSegmentationResult,
)
from open_vocabulary_segmentation.array_types import BoolArray, Float32Array
from open_vocabulary_segmentation.result import MaskBounds, MaskSuppression

IMAGE_SIZE: ImageSize = ImageSize(width=20, height=10)


def build_logits() -> MaskLogits:
    return MaskLogits(
        logits=np.zeros((4, 4), dtype=np.float32),
        backend=PromptableBackend.SAM2,
        weights_path="w",
        image_size=IMAGE_SIZE,
    )


def build_result() -> ObjectSegmentationResult:
    masks: BoolArray = np.zeros((4, IMAGE_SIZE.height, IMAGE_SIZE.width), dtype=np.bool_)
    masks[0, 0:5, 0:5] = True
    masks[1, 0:5, 0:6] = True
    masks[2, 5:10, 10:20] = True
    return ObjectSegmentationResult(
        masks=masks,
        scores=np.array([0.6, 0.9, 0.7, 0.2]),
        stability_scores=np.array([0.95, 0.8, 0.99, 0.0]),
        prompt_indices=np.array([0, 0, 1, 2]),
        prompts=(
            ObjectPrompt.from_coordinates(positive_points=[(2, 2)]),
            ObjectPrompt.from_coordinates(box_xyxy=(10, 5, 20, 10)),
            ObjectPrompt.from_coordinates(positive_points=[(19, 0)]),
        ),
        mask_logits=(build_logits(), build_logits(), build_logits(), None),
        image_size=IMAGE_SIZE,
    )


def test_items_expose_boxes_prompts_and_emptiness() -> None:
    result: ObjectSegmentationResult = build_result()
    first: ObjectSegmentation = result[0]
    assert first.box is not None
    assert first.box.value.tolist() == [0, 0, 5, 5]
    assert first.area == 25
    assert first.prompt_index == 0
    assert result[-1].is_empty
    assert result[-1].box is None
    assert result.pixel_xyxy[3].tolist() == [0, 0, 0, 0]
    assert result.is_empty_mask.tolist() == [False, False, False, True]
    assert [segmentation.prompt_index for segmentation in result] == [0, 0, 1, 2]
    with pytest.raises(IndexError):
        result[4]


def test_selection_and_filters_keep_every_prompt() -> None:
    result: ObjectSegmentationResult = build_result()
    assert result.filter_by_score(0.65).scores.tolist() == pytest.approx([0.9, 0.7])
    assert result.filter_by_stability(0.9).prompt_indices.tolist() == [0, 1]
    assert len(result.filter_by_area(1)) == 3
    assert result.sort_by_score().scores.tolist() == pytest.approx([0.9, 0.7, 0.6, 0.2])
    assert result.best_per_prompt().scores.tolist() == pytest.approx([0.9, 0.7, 0.2])
    assert result.of_prompt(0).scores.tolist() == pytest.approx([0.6, 0.9])
    assert len(result.select(np.array([2])).prompts) == 3
    with pytest.raises(IndexError):
        result.of_prompt(3)


def test_nms_union_and_instance_map() -> None:
    result: ObjectSegmentationResult = build_result()
    suppressed: ObjectSegmentationResult = result.non_maximum_suppression(0.5)
    assert suppressed.scores.tolist() == pytest.approx([0.9, 0.7, 0.2])
    assert int(result.union_mask().value.sum()) == 30 + 50
    instance_map = result.instance_map()
    assert int(instance_map[0, 0]) == 1
    assert int(instance_map[9, 0]) == -1


def test_refine_without_logits_keeps_the_prompt_mask() -> None:
    result: ObjectSegmentationResult = build_result()
    refined: ObjectPrompt = result[3].refine(positive_points=Points2D(value=np.array([[1.0, 1.0]])))
    assert refined.mask is None
    assert refined.point_count == 2
    assert isinstance(result[0].refine().mask, MaskLogits)


def test_invalid_results_raise() -> None:
    result: ObjectSegmentationResult = build_result()
    with pytest.raises(ValueError, match="scores must be in"):
        ObjectSegmentationResult(
            masks=result.masks,
            scores=np.array([1.5, 0.9, 0.7, 0.2]),
            stability_scores=result.stability_scores,
            prompt_indices=result.prompt_indices,
            prompts=result.prompts,
            mask_logits=result.mask_logits,
            image_size=IMAGE_SIZE,
        )
    with pytest.raises(ValueError, match="prompt_indices must be in"):
        ObjectSegmentationResult(
            masks=result.masks,
            scores=result.scores,
            stability_scores=result.stability_scores,
            prompt_indices=np.array([0, 0, 1, 5]),
            prompts=result.prompts,
            mask_logits=result.mask_logits,
            image_size=IMAGE_SIZE,
        )
    with pytest.raises(ValueError, match="mask_logits must have 4 entries"):
        ObjectSegmentationResult(
            masks=result.masks,
            scores=result.scores,
            stability_scores=result.stability_scores,
            prompt_indices=result.prompt_indices,
            prompts=result.prompts,
            mask_logits=(),
            image_size=IMAGE_SIZE,
        )
    empty: ObjectSegmentationResult = ObjectSegmentationResult.empty(result.prompts, IMAGE_SIZE)
    assert len(empty) == 0 and len(empty.prompts) == 3


def test_mask_bounds_and_suppression_helpers() -> None:
    masks: BoolArray = np.zeros((3, 4, 4), dtype=np.bool_)
    masks[0, 1:3, 1:4] = True
    masks[1, 1:3, 1:4] = True
    assert MaskBounds.pixel_xyxy(masks).tolist() == [[1, 1, 4, 3], [1, 1, 4, 3], [0, 0, 0, 0]]
    kept = MaskSuppression.kept_indices(
        masks, MaskBounds.pixel_xyxy(masks), np.array([0.2, 0.9, 0.5]), 0.5, group_ids=np.array([0, 1, 0])
    )
    assert kept.tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match="iou_threshold"):
        MaskSuppression.kept_indices(masks, MaskBounds.pixel_xyxy(masks), np.zeros(3), 1.5)


def test_semantic_result_maps_classes() -> None:
    probabilities: Float32Array = np.zeros((2, 2, 3), dtype=np.float32)
    probabilities[0, 0, 0] = 0.9
    probabilities[1, 0, 0] = 0.6
    probabilities[1, 1, 2] = 0.7
    semantic: SemanticSegmentationResult = SemanticSegmentationResult(
        class_probabilities=probabilities,
        prompt=Prompt.from_class_names(("cat", "dog")),
        image_size=ImageSize(width=3, height=2),
    )
    assert semantic.class_map().tolist() == [[0, -1, -1], [-1, -1, 1]]
    assert semantic.class_mask("dog", 0.5).value.tolist() == [[True, False, False], [False, False, True]]
    assert float(semantic.probability_of(" cat ")[0, 0]) == pytest.approx(0.9)
    with pytest.raises(KeyError):
        semantic.probability_of("bird")
    with pytest.raises(ValueError, match="threshold"):
        semantic.class_map(1.5)
    with pytest.raises(ValueError, match="class_probabilities must have shape"):
        SemanticSegmentationResult(
            class_probabilities=probabilities[:1],
            prompt=Prompt.from_class_names(("cat", "dog")),
            image_size=ImageSize(width=3, height=2),
        )
