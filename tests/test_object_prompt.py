import numpy as np
import pytest
import torch
from geometry import Box2D, Box2DFormat, Points2D
from image_container import BinaryImage

from open_vocabulary_segmentation import (
    Device,
    ImageSize,
    MaskLogits,
    MaskOutput,
    ObjectPrompt,
    Precision,
    PromptableBackend,
    PromptElement,
)
from open_vocabulary_segmentation.array_types import BoolArray, Float32Array
from open_vocabulary_segmentation.backends.mask_decoder import DecodedMasks, PointLabel, PromptSignature, PromptTensors
from open_vocabulary_segmentation.promptable import ModelFrame
from open_vocabulary_segmentation.runtime import TorchRuntime

IMAGE_SIZE: ImageSize = ImageSize(width=40, height=20)
RUNTIME: TorchRuntime = TorchRuntime(Device.CPU, Precision.FLOAT32)


def test_from_coordinates_builds_points_and_box() -> None:
    prompt: ObjectPrompt = ObjectPrompt.from_coordinates(
        positive_points=[(1, 2), (3, 4)], negative_points=[(5, 6)], box_xyxy=(0, 0, 10, 10)
    )
    assert prompt.point_xy.tolist() == [[1, 2], [3, 4], [5, 6]]
    assert prompt.is_point_positive.tolist() == [True, True, False]
    assert prompt.point_count == 3
    assert prompt.elements == frozenset({PromptElement.POINT, PromptElement.BOX})
    assert prompt.xyxy is not None
    assert prompt.xyxy.tolist() == [0, 0, 10, 10]
    assert not prompt.is_ambiguous


def test_boxes_of_any_format_are_read_as_xyxy() -> None:
    prompt: ObjectPrompt = ObjectPrompt(
        box=Box2D.register(value=np.array([2.0, 3.0, 4.0, 5.0]), box2d_format=Box2DFormat.XYWH)
    )
    assert prompt.xyxy is not None
    assert prompt.xyxy.tolist() == [2, 3, 6, 8]
    assert prompt.point_xy.shape == (0, 2)


def test_single_point_is_ambiguous() -> None:
    assert ObjectPrompt.from_coordinates(positive_points=[(1, 1)]).is_ambiguous
    assert not ObjectPrompt.from_coordinates(positive_points=[(1, 1)], box_xyxy=(0, 0, 4, 4)).is_ambiguous


def test_invalid_prompts_raise() -> None:
    with pytest.raises(ValueError, match="needs positive points, a box or a mask"):
        ObjectPrompt.from_coordinates(negative_points=[(1, 1)])
    with pytest.raises(ValueError, match="at least one point"):
        ObjectPrompt(positive_points=Points2D(value=np.zeros((0, 2))))
    with pytest.raises(ValueError, match="finite"):
        ObjectPrompt.from_coordinates(positive_points=[(float("nan"), 1.0)])
    with pytest.raises(ValueError, match="x1 must be less than x2"):
        ObjectPrompt.from_coordinates(box_xyxy=(5, 5, 5, 9))


def test_validate_for_checks_points_box_and_mask() -> None:
    ObjectPrompt.from_coordinates(positive_points=[(40, 20)], box_xyxy=(0, 0, 40, 20)).validate_for(IMAGE_SIZE)
    with pytest.raises(ValueError, match="points must lie inside"):
        ObjectPrompt.from_coordinates(positive_points=[(41, 0)]).validate_for(IMAGE_SIZE)
    with pytest.raises(ValueError, match="box must lie inside"):
        ObjectPrompt.from_coordinates(box_xyxy=(-1, 0, 10, 10)).validate_for(IMAGE_SIZE)
    with pytest.raises(ValueError, match="mask must have shape"):
        ObjectPrompt(mask=BinaryImage(value=np.zeros((10, 10), dtype=np.bool_))).validate_for(IMAGE_SIZE)


def test_with_points_and_with_mask_extend_the_prompt() -> None:
    prompt: ObjectPrompt = ObjectPrompt.from_coordinates(positive_points=[(1, 1)], box_xyxy=(0, 0, 9, 9))
    extended: ObjectPrompt = prompt.with_points(negative_points=Points2D(value=np.array([[2.0, 2.0]])))
    assert extended.point_xy.tolist() == [[1, 1], [2, 2]]
    assert extended.box is prompt.box
    mask: BinaryImage = BinaryImage(value=np.zeros((20, 40), dtype=np.bool_))
    assert extended.with_mask(mask).mask is mask
    assert extended.with_mask(None).mask is None


def test_mask_logits_compatibility_and_validation() -> None:
    logits: MaskLogits = MaskLogits(
        logits=np.zeros((4, 4), dtype=np.float64).astype(np.float32),
        backend=PromptableBackend.SAM2,
        weights_path="w",
        image_size=IMAGE_SIZE,
    )
    assert logits.logits.dtype == np.float32
    assert logits.is_compatible_with(PromptableBackend.SAM2, "w", IMAGE_SIZE)
    assert not logits.is_compatible_with(PromptableBackend.SAM2, "other", IMAGE_SIZE)
    assert not logits.is_compatible_with(PromptableBackend.SAM2, "w", ImageSize(width=1, height=1))
    with pytest.raises(ValueError, match="shape"):
        MaskLogits(
            logits=np.zeros(4, dtype=np.float32), backend=PromptableBackend.SAM, weights_path="w", image_size=IMAGE_SIZE
        )
    with pytest.raises(ValueError, match="finite"):
        MaskLogits(
            logits=np.full((2, 2), np.inf, dtype=np.float32),
            backend=PromptableBackend.SAM,
            weights_path="w",
            image_size=IMAGE_SIZE,
        )


def test_padded_frame_maps_prompts_and_masks() -> None:
    frame: ModelFrame = ModelFrame(image_size=IMAGE_SIZE, resized_size=ImageSize(width=16, height=8), input_size=16)
    assert frame.scale_xy.tolist() == pytest.approx([0.4, 0.4])
    assert frame.to_model_xy(np.array([[10.0, 10.0]]))[0].tolist() == pytest.approx([4.0, 4.0])
    assert frame.to_model_xyxy(np.array([0.0, 0.0, 40.0, 20.0])).tolist() == pytest.approx([0, 0, 16, 8])
    mask: BoolArray = np.ones((20, 40), dtype=np.bool_)
    logits: Float32Array = frame.to_mask_logits(mask, 8)
    assert logits.shape == (8, 8)
    assert float(logits[0, 0]) == pytest.approx(ModelFrame.MASK_PROMPT_LOGIT)
    assert float(logits[7, 0]) == pytest.approx(-ModelFrame.MASK_PROMPT_LOGIT)
    with pytest.raises(ValueError, match="must fit"):
        ModelFrame(image_size=IMAGE_SIZE, resized_size=ImageSize(width=20, height=8), input_size=16)


def test_prompt_signatures_group_prompts() -> None:
    point: PromptSignature = PromptSignature.of(ObjectPrompt.from_coordinates(positive_points=[(1, 1)]))
    assert point.is_ambiguous and point.is_batchable
    masked: PromptSignature = PromptSignature.of(
        ObjectPrompt(mask=BinaryImage(value=np.zeros((20, 40), dtype=np.bool_)))
    )
    assert not masked.is_batchable
    assert point != PromptSignature.of(ObjectPrompt.from_coordinates(positive_points=[(1, 1), (2, 2)]))


def test_prompt_tensors_layout() -> None:
    frame: ModelFrame = ModelFrame(image_size=IMAGE_SIZE, resized_size=ImageSize(width=16, height=8), input_size=16)
    prompts: list[ObjectPrompt] = [
        ObjectPrompt.from_coordinates(positive_points=[(10, 10)], negative_points=[(20, 5)], box_xyxy=(0, 0, 40, 20)),
        ObjectPrompt.from_coordinates(positive_points=[(5, 5)], negative_points=[(6, 6)], box_xyxy=(0, 0, 20, 10)),
    ]
    tensors: PromptTensors = PromptTensors.build(prompts, frame, 4, RUNTIME)
    assert tensors.points is not None and tensors.labels is not None and tensors.boxes is not None
    assert tuple(tensors.points.shape) == (1, 2, 2, 2)
    assert tensors.labels[0, 0].tolist() == [PointLabel.POSITIVE, PointLabel.NEGATIVE]
    assert tensors.boxes[0, 1].tolist() == pytest.approx([0, 0, 8, 4])
    assert tensors.masks is None
    mask_only: PromptTensors = PromptTensors.build(
        [ObjectPrompt(mask=BinaryImage(value=np.ones((20, 40), dtype=np.bool_)))], frame, 4, RUNTIME
    )
    assert mask_only.points is not None and mask_only.labels is not None and mask_only.masks is not None
    assert mask_only.labels.tolist() == [[[PointLabel.NOT_A_POINT]]]
    assert tuple(mask_only.masks.shape) == (1, 1, 4, 4)
    box_only: PromptTensors = PromptTensors.build(
        [ObjectPrompt.from_coordinates(box_xyxy=(0, 0, 4, 4))], frame, 4, RUNTIME
    )
    assert box_only.points is None
    with pytest.raises(ValueError, match="share one signature"):
        PromptTensors.build([prompts[0], ObjectPrompt.from_coordinates(box_xyxy=(0, 0, 4, 4))], frame, 4, RUNTIME)


def test_decoded_masks_keep_the_best_candidate() -> None:
    decoded: DecodedMasks = DecodedMasks(
        low_resolution_logits=torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 2, 2),
        iou_scores=torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.1, 0.3]]),
    )
    best: DecodedMasks = decoded.best()
    assert best.masks_per_prompt == 1
    assert best.iou_scores[:, 0].tolist() == pytest.approx([0.9, 0.8])
    assert torch.equal(best.low_resolution_logits[0, 0], decoded.low_resolution_logits[0, 1])
    with pytest.raises(ValueError, match="iou_scores must have shape"):
        DecodedMasks(low_resolution_logits=torch.zeros(1, 3, 2, 2), iou_scores=torch.zeros(1, 1))


def test_mask_output_chooses_the_head() -> None:
    assert MaskOutput.AUTO.is_multimask(is_ambiguous=True)
    assert not MaskOutput.AUTO.is_multimask(is_ambiguous=False)
    assert not MaskOutput.SINGLE.is_multimask(is_ambiguous=True)
    assert MaskOutput.BEST_OF_MULTIPLE.is_multimask(is_ambiguous=False)
    assert MaskOutput.ALL_CANDIDATES.is_every_candidate_kept
    assert not MaskOutput.BEST_OF_MULTIPLE.is_every_candidate_kept
