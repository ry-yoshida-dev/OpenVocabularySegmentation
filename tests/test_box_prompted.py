from collections.abc import Sequence
from typing import ClassVar, cast

import numpy as np
import pytest
import torch
from open_vocabulary_detector import DetectionResult, OpenVocabularyDetector
from PIL import Image
from transformers import BatchEncoding

from open_vocabulary_segmentation import (
    DetectionThresholds,
    DetectorBackend,
    DetectorSettings,
    Device,
    ImageSize,
    Prompt,
    PromptKind,
    SegmentationResult,
    SegmentationThresholds,
    SegmenterBackend,
    SegmenterSettings,
    TextQuery,
)
from open_vocabulary_segmentation.array_types import BoolArray, FloatArray
from open_vocabulary_segmentation.backends.box_prompted import BoxPromptedSegmenter
from open_vocabulary_segmentation.backends.checkpoint_config import CheckpointConfig

IMAGE_BOXES: dict[int, list[list[float]]] = {
    100: [[10.0, 10.0, 30.0, 20.0], [50.0, 5.0, 90.0, 45.0]],
    80: [],
    60: [[0.0, 0.0, 6.0, 6.0]],
}
IMAGE_CONFIDENCES: dict[int, list[float]] = {100: [0.9, 0.8], 80: [], 60: [0.7]}
IMAGE_QUERY_IDS: dict[int, list[int]] = {100: [1, 2], 80: [], 60: [0]}


class StubBoxDetector(OpenVocabularyDetector):
    BACKEND: ClassVar[DetectorBackend] = DetectorBackend.OWL_VIT

    def _detect_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[DetectionResult]:
        return [
            DetectionResult.from_xyxy(
                xyxy=np.array(IMAGE_BOXES[image.width], dtype=np.float64).reshape(-1, 4),
                confidences=np.array(IMAGE_CONFIDENCES[image.width], dtype=np.float64),
                query_ids=np.array(IMAGE_QUERY_IDS[image.width], dtype=np.int64),
                prompt=prompt,
                image_size=ImageSize.from_image(image),
            )
            for image in images
        ]


class StubBoxPromptedSegmenter(BoxPromptedSegmenter[torch.Tensor]):
    BACKEND: ClassVar[SegmenterBackend] = SegmenterBackend.SAM
    MODEL_TYPES: ClassVar[frozenset[str]] = frozenset({"sam"})
    BOX_CHUNK_SIZE: ClassVar[int] = 1

    def __init__(self, settings: SegmenterSettings) -> None:
        super().__init__(settings)
        self.embedded_batch_sizes: list[int] = []
        self.decoded_box_counts: list[int] = []

    def _preprocess(self, image: Image.Image, xyxy: FloatArray) -> BatchEncoding:
        return BatchEncoding(
            {
                "pixel_values": torch.zeros(1, 3, 4, 4),
                "input_boxes": torch.from_numpy(xyxy).float()[None],
                "image_size": torch.tensor([image.height, image.width]),
            }
        )

    def _embed_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.embedded_batch_sizes.append(int(pixel_values.shape[0]))
        return pixel_values

    def _decode_masks(self, embeddings: torch.Tensor, image_index: int, input_boxes: torch.Tensor) -> torch.Tensor:
        self.decoded_box_counts.append(int(input_boxes.shape[1]))
        return input_boxes[:, :, None, None, :]

    def _upscale_masks(self, mask_logits: torch.Tensor, inputs: BatchEncoding) -> BoolArray:
        height, width = inputs["image_size"].tolist()
        masks: BoolArray = np.zeros((mask_logits.shape[1], height, width), dtype=np.bool_)
        boxes: list[list[int]] = cast(list[list[int]], mask_logits[0, :, 0, 0].round().long().tolist())
        for index, (x_min, y_min, x_max, y_max) in enumerate(boxes):
            masks[index, y_min:y_max, x_min:x_max] = True
        return masks


def build_stub_box_detector(settings: DetectorSettings) -> StubBoxDetector:
    return StubBoxDetector(settings)


def patch_checkpoint(monkeypatch: pytest.MonkeyPatch, model_type: str) -> None:
    def read_model_type(weights_path: str) -> str:
        return model_type

    monkeypatch.setattr(CheckpointConfig, "read_model_type", staticmethod(read_model_type))
    monkeypatch.setattr(DetectorSettings, "build", build_stub_box_detector)


@pytest.fixture
def segmenter(monkeypatch: pytest.MonkeyPatch) -> StubBoxPromptedSegmenter:
    patch_checkpoint(monkeypatch, "sam")
    return StubBoxPromptedSegmenter(build_settings())


def build_settings() -> SegmenterSettings:
    return SegmenterSettings(
        backend=SegmenterBackend.SAM,
        weights_path="stub",
        thresholds=SegmentationThresholds(confidence_threshold=0.0, nms_iou_threshold=None),
        box_detector=DetectorSettings(
            backend=DetectorBackend.OWL_VIT,
            weights_path="stub",
            thresholds=DetectionThresholds(confidence_threshold=0.1, nms_iou_threshold=None),
            device=Device.CPU,
        ),
        batch_size=3,
        device=Device.CPU,
    )


def test_boxes_become_masks_with_detector_confidences_and_queries(segmenter: StubBoxPromptedSegmenter) -> None:
    prompt: Prompt = Prompt.from_texts({"car": ("car", "suv"), "dog": ("dog",)})
    images: list[Image.Image] = [Image.new("RGB", (width, 50)) for width in (100, 80, 60)]
    results: list[SegmentationResult] = segmenter.segment_images(images, prompt)
    assert results[0].xyxy.tolist() == IMAGE_BOXES[100]
    assert results[0].confidences.tolist() == pytest.approx([0.9, 0.8])
    assert [segmentation.matched_query for segmentation in results[0]] == [TextQuery("suv"), TextQuery("dog")]
    assert len(results[1]) == 0
    assert results[2].xyxy.tolist() == IMAGE_BOXES[60]
    assert results[2].class_names == ("car", "dog")


def test_images_without_boxes_skip_the_model_and_boxes_are_chunked(segmenter: StubBoxPromptedSegmenter) -> None:
    images: list[Image.Image] = [Image.new("RGB", (width, 50)) for width in (100, 80, 60)]
    segmenter.segment_images(images, Prompt.from_class_names(("car", "suv", "dog")))
    assert segmenter.embedded_batch_sizes == [2]
    assert segmenter.decoded_box_counts == [1, 1, 1]


def test_mini_batch_without_boxes_runs_no_model(segmenter: StubBoxPromptedSegmenter) -> None:
    results: list[SegmentationResult] = segmenter.segment_images(
        [Image.new("RGB", (80, 50))], Prompt.from_class_names(("car",))
    )
    assert len(results[0]) == 0
    assert segmenter.embedded_batch_sizes == []


def test_query_kinds_follow_the_box_detector(segmenter: StubBoxPromptedSegmenter, mixed_prompt: Prompt) -> None:
    assert segmenter.supported_prompt_kinds == frozenset({PromptKind.TEXT, PromptKind.VISUAL})
    assert isinstance(segmenter.box_detector, StubBoxDetector)
    results: list[SegmentationResult] = segmenter.segment_images([Image.new("RGB", (60, 50))], mixed_prompt)
    assert len(results[0]) == 1


def test_checkpoint_of_another_architecture_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_checkpoint(monkeypatch, "sam_hq")
    with pytest.raises(ValueError, match="is a 'sam_hq' checkpoint"):
        StubBoxPromptedSegmenter(build_settings())
