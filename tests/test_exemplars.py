import numpy as np
import pytest
from geometry import Box2DFormat, Boxes2D

from open_vocabulary_segmentation import (
    BoxExemplars,
    Device,
    ImageExemplars,
    ImageSize,
    Precision,
    Prompt,
    SegmentationResult,
)
from open_vocabulary_segmentation.array_types import BoolArray
from open_vocabulary_segmentation.backends.sam3.exemplar_tensors import ExemplarTensors
from open_vocabulary_segmentation.backends.sam3.prediction import Sam3Prediction
from open_vocabulary_segmentation.runtime import TorchRuntime

IMAGE_SIZE: ImageSize = ImageSize(width=100, height=50)


def boxes(xyxy: list[list[float]], box_format: Box2DFormat = Box2DFormat.XYXY) -> Boxes2D:
    return Boxes2D.register(value=np.array(xyxy, dtype=np.float64), box2d_format=box_format)


def test_box_exemplars_order_positive_boxes_first() -> None:
    exemplars: BoxExemplars = BoxExemplars(
        positive_boxes=boxes([[0, 0, 10, 10]]),
        negative_boxes=boxes([[20, 20, 5, 5]], Box2DFormat.XYWH),
    )
    assert exemplars.xyxy.tolist() == [[0, 0, 10, 10], [20, 20, 25, 25]]
    assert exemplars.is_positive.tolist() == [True, False]
    exemplars.validate_for(IMAGE_SIZE)
    with pytest.raises(ValueError, match="inside the 10x10 image"):
        exemplars.validate_for(ImageSize(width=10, height=10))


def test_invalid_box_exemplars_raise() -> None:
    with pytest.raises(ValueError, match="positive or negative boxes"):
        BoxExemplars()
    with pytest.raises(ValueError, match="positive box to define the class"):
        BoxExemplars(negative_boxes=boxes([[0, 0, 5, 5]]), is_text_ignored=True)
    with pytest.raises(ValueError, match="at least one box"):
        BoxExemplars(positive_boxes=Boxes2D.register(value=np.zeros((0, 4)), box2d_format=Box2DFormat.XYXY))


def test_image_exemplars_are_keyed_by_stripped_class_names() -> None:
    exemplars: BoxExemplars = BoxExemplars(positive_boxes=boxes([[0, 0, 5, 5]]))
    image_exemplars: ImageExemplars = ImageExemplars({" car ": exemplars})
    assert image_exemplars.of_class("car") is exemplars
    assert image_exemplars.of_class("dog") is None
    image_exemplars.validate_for(Prompt.from_class_names(("car",)), IMAGE_SIZE)
    with pytest.raises(KeyError, match="'car' is not a class"):
        image_exemplars.validate_for(Prompt.from_class_names(("dog",)), IMAGE_SIZE)
    with pytest.raises(ValueError, match="unique"):
        ImageExemplars({"car": exemplars, " car": exemplars})
    with pytest.raises(ValueError, match="at least one class"):
        ImageExemplars({})


def test_exemplar_tensors_are_normalized_cxcywh_and_padded() -> None:
    first: BoxExemplars = BoxExemplars(positive_boxes=boxes([[0, 0, 50, 50]]), negative_boxes=boxes([[50, 0, 100, 50]]))
    second: BoxExemplars = BoxExemplars(positive_boxes=boxes([[0, 0, 10, 10]]))
    tensors: ExemplarTensors = ExemplarTensors.build(
        [first, second], [IMAGE_SIZE, ImageSize(width=20, height=20)], TorchRuntime(Device.CPU, Precision.FLOAT32)
    )
    assert tuple(tensors.boxes.shape) == (2, 2, 4)
    assert tensors.boxes[0, 0].tolist() == pytest.approx([0.25, 0.5, 0.5, 1.0])
    assert tensors.boxes[1, 0].tolist() == pytest.approx([0.25, 0.25, 0.5, 0.5])
    assert tensors.labels.tolist() == [[1, 0], [1, ExemplarTensors.PADDING_LABEL]]
    with pytest.raises(ValueError, match="expected exemplars for 1 images"):
        ExemplarTensors.build([first, second], [IMAGE_SIZE], TorchRuntime(Device.CPU, Precision.FLOAT32))


def test_sam3_prediction_merges_query_runs() -> None:
    prompt: Prompt = Prompt.from_texts({"car": ("car", "suv"), "dog": ("dog",)})
    prediction: Sam3Prediction = Sam3Prediction.start(prompt, [IMAGE_SIZE], is_semantic_requested=True)
    masks: BoolArray = np.zeros((1, 50, 100), dtype=np.bool_)
    masks[0, :10, :10] = True
    prediction.add_instances(0, masks, np.array([0.8]), np.array([1]))
    prediction.add_instances(0, np.zeros((0, 50, 100), dtype=np.bool_), np.zeros(0), np.zeros(0, dtype=np.int64))
    prediction.add_semantic(0, 0, np.full((50, 100), 0.2, dtype=np.float32))
    prediction.add_semantic(0, 0, np.full((50, 100), 0.6, dtype=np.float32))
    instances: SegmentationResult = prediction.instances[0]
    assert instances.class_names == ("car", "dog")
    assert instances.query_ids.tolist() == [1]
    probabilities = prediction.semantics[0].class_probabilities
    assert float(probabilities[0, 0, 0]) == pytest.approx(0.6)
    assert float(probabilities[1, 0, 0]) == pytest.approx(0.0)
    plain: Sam3Prediction = Sam3Prediction.start(prompt, [IMAGE_SIZE], is_semantic_requested=False)
    assert len(plain.instances[0]) == 0
    with pytest.raises(ValueError, match="not requested"):
        plain.add_semantic(0, 0, np.zeros((50, 100), dtype=np.float32))
