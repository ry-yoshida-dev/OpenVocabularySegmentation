from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest
from open_vocabulary_detector import DetectionResult, OpenVocabularyDetector
from PIL import Image
from stub_promptable import StubPromptableSegmenter, patch_checkpoint

from open_vocabulary_segmentation import (
    DetectionThresholds,
    DetectorBackend,
    DetectorSettings,
    Device,
    ImageSize,
    Prompt,
    PromptableSegmenter,
    PromptableSegmenterSettings,
    PromptKind,
    SegmentationResult,
    SegmentationThresholds,
    SegmenterBackend,
    SegmenterSettings,
    TextQuery,
)
from open_vocabulary_segmentation.backends.box_prompted import BoxPromptedSegmenter

IMAGE_BOXES: dict[int, list[list[float]]] = {
    40: [[-10.0, -10.0, 20.0, 20.0], [50.0, 0.0, 60.0, 10.0]],
    100: [[10.0, 10.0, 30.0, 20.0], [50.0, 5.0, 90.0, 45.0]],
    80: [],
    60: [[0.0, 0.0, 6.0, 6.0]],
}
STUB_RESOLUTION_ERROR: float = 7.0
IMAGE_CONFIDENCES: dict[int, list[float]] = {40: [0.9, 0.8], 100: [0.9, 0.8], 80: [], 60: [0.7]}
IMAGE_QUERY_IDS: dict[int, list[int]] = {40: [0, 1], 100: [1, 2], 80: [], 60: [0]}


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


class StubBoxPromptedSegmenter(BoxPromptedSegmenter):
    def _build_mask_segmenter(self, settings: PromptableSegmenterSettings) -> PromptableSegmenter:
        return StubPromptableSegmenter(settings)

    @property
    def stub_mask_segmenter(self) -> StubPromptableSegmenter:
        mask_segmenter: PromptableSegmenter = self.mask_segmenter
        assert isinstance(mask_segmenter, StubPromptableSegmenter)
        return mask_segmenter


def build_stub_box_detector(settings: DetectorSettings) -> StubBoxDetector:
    return StubBoxDetector(settings)


@pytest.fixture
def segmenter(monkeypatch: pytest.MonkeyPatch) -> StubBoxPromptedSegmenter:
    patch_checkpoint(monkeypatch, "sam2_video")
    monkeypatch.setattr(DetectorSettings, "build", build_stub_box_detector)
    return StubBoxPromptedSegmenter(build_settings())


def build_settings() -> SegmenterSettings:
    return SegmenterSettings(
        backend=SegmenterBackend.SAM2,
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
    assert np.abs(results[0].xyxy - np.array(IMAGE_BOXES[100])).max() <= STUB_RESOLUTION_ERROR
    assert results[0].confidences.tolist() == pytest.approx([0.9, 0.8])
    assert [segmentation.matched_query for segmentation in results[0]] == [TextQuery("suv"), TextQuery("dog")]
    assert len(results[1]) == 0
    assert np.abs(results[2].xyxy - np.array(IMAGE_BOXES[60])).max() <= STUB_RESOLUTION_ERROR
    assert results[2].class_names == ("car", "dog")


def test_images_without_boxes_skip_the_model_and_boxes_share_a_pass(segmenter: StubBoxPromptedSegmenter) -> None:
    images: list[Image.Image] = [Image.new("RGB", (width, 50)) for width in (100, 80, 60)]
    segmenter.segment_images(images, Prompt.from_class_names(("car", "suv", "dog")))
    assert segmenter.stub_mask_segmenter.embedded_batch_sizes == [2]
    assert segmenter.stub_mask_segmenter.decoded_prompt_counts == [2, 1]
    assert segmenter.stub_mask_segmenter.multimask_flags == [False, False]


def test_mini_batch_without_boxes_runs_no_model(segmenter: StubBoxPromptedSegmenter) -> None:
    results: list[SegmentationResult] = segmenter.segment_images(
        [Image.new("RGB", (80, 50))], Prompt.from_class_names(("car",))
    )
    assert len(results[0]) == 0
    assert segmenter.stub_mask_segmenter.embedded_batch_sizes == []


def test_query_kinds_follow_the_box_detector(segmenter: StubBoxPromptedSegmenter, mixed_prompt: Prompt) -> None:
    assert segmenter.supported_prompt_kinds == frozenset({PromptKind.TEXT, PromptKind.VISUAL})
    assert isinstance(segmenter.box_detector, StubBoxDetector)
    results: list[SegmentationResult] = segmenter.segment_images([Image.new("RGB", (60, 50))], mixed_prompt)
    assert len(results[0]) == 1


def test_boxes_are_clipped_to_the_image_and_degenerate_boxes_dropped(segmenter: StubBoxPromptedSegmenter) -> None:
    results: list[SegmentationResult] = segmenter.segment_images(
        [Image.new("RGB", (40, 50))], Prompt.from_class_names(("car", "suv", "dog"))
    )
    assert len(results[0]) == 1
    assert results[0].pixel_xyxy[0, :2].tolist() == [0, 0]


def test_checkpoint_of_another_architecture_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_checkpoint(monkeypatch, "sam_hq")
    monkeypatch.setattr(DetectorSettings, "build", build_stub_box_detector)
    with pytest.raises(ValueError, match="is a 'sam_hq' checkpoint"):
        StubBoxPromptedSegmenter(build_settings())
