from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest
from image_container import ChannelOrder, ImageContainer
from PIL import Image

from open_vocabulary_segmentation import (
    Device,
    ImageSize,
    OpenVocabularySegmenter,
    Prompt,
    PromptKind,
    SegmentationResult,
    SegmentationThresholds,
    SegmenterBackend,
    SegmenterSettings,
    TextQuery,
)
from open_vocabulary_segmentation.array_types import BoolArray

CONFIDENCES: list[float] = [0.7, 0.9, 0.1, 0.6]
QUERY_IDS: list[int] = [0, 0, 0, 1]


def build_masks(image_size: ImageSize) -> BoolArray:
    masks: BoolArray = np.zeros((4, image_size.height, image_size.width), dtype=np.bool_)
    masks[0, 10:30, 10:30] = True
    masks[1, 11:31, 11:31] = True
    masks[2, 0:5, 0:5] = True
    masks[3, 35:45, 60:80] = True
    return masks


class StubSegmenter(OpenVocabularySegmenter):
    BACKEND: ClassVar[SegmenterBackend] = SegmenterBackend.SAM3

    def __init__(self, settings: SegmenterSettings) -> None:
        super().__init__(settings)
        self.mini_batch_sizes: list[int] = []
        self.query_ids: list[int] = QUERY_IDS
        self.received_modes: list[str] = []

    @property
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        return frozenset({PromptKind.TEXT})

    def _segment_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[SegmentationResult]:
        self.mini_batch_sizes.append(len(images))
        self.received_modes.extend(image.mode for image in images)
        return [
            SegmentationResult.from_masks(
                masks=build_masks(ImageSize.from_image(image)),
                confidences=np.array(CONFIDENCES, dtype=np.float64),
                query_ids=np.array(self.query_ids, dtype=np.int64),
                prompt=prompt,
                image_size=ImageSize.from_image(image),
            )
            for image in images
        ]


class TruncatingSegmenter(StubSegmenter):
    def _segment_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[SegmentationResult]:
        return super()._segment_mini_batch(images, prompt)[:-1]


def build_settings(nms_iou_threshold: float | None = None) -> SegmenterSettings:
    return SegmenterSettings(
        backend=SegmenterBackend.SAM3,
        weights_path="stub",
        thresholds=SegmentationThresholds(confidence_threshold=0.5, nms_iou_threshold=nms_iou_threshold),
        batch_size=2,
        device=Device.CPU,
    )


def test_segment_images_thresholds_sorts_and_batches() -> None:
    segmenter: StubSegmenter = StubSegmenter(build_settings())
    images: list[Image.Image] = [Image.new("L", (100, 50))] * 5
    results: list[SegmentationResult] = segmenter.segment_images(images, Prompt.from_class_names(("cat", "dog")))
    assert segmenter.mini_batch_sizes == [2, 2, 1]
    assert set(segmenter.received_modes) == {"RGB"}
    assert len(results) == 5
    assert results[0].confidences.tolist() == pytest.approx([0.9, 0.7, 0.6])
    assert results[0].class_ids.tolist() == [0, 0, 1]
    assert results[0].image_size == ImageSize(width=100, height=50)


def test_segment_applies_class_wise_mask_nms() -> None:
    segmenter: StubSegmenter = StubSegmenter(build_settings(nms_iou_threshold=0.5))
    result: SegmentationResult = segmenter.segment(Image.new("RGB", (100, 50)), Prompt.from_class_names(("cat", "dog")))
    assert result.confidences.tolist() == pytest.approx([0.9, 0.6])


def test_queries_of_one_class_are_merged_by_class_wise_nms() -> None:
    segmenter: StubSegmenter = StubSegmenter(build_settings(nms_iou_threshold=0.5))
    segmenter.query_ids = [1, 2, 0, 3]
    prompt: Prompt = Prompt.from_texts({"car": ("car", "suv", "taxi"), "dog": ("dog",)})
    result: SegmentationResult = segmenter.segment(Image.new("RGB", (100, 50)), prompt)
    assert [(segmentation.class_name, segmentation.matched_query) for segmentation in result] == [
        ("car", TextQuery("taxi")),
        ("dog", TextQuery("dog")),
    ]


def test_empty_image_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        StubSegmenter(build_settings()).segment_images([], Prompt.from_class_names(("cat",)))


def test_unsupported_query_kind_is_rejected_before_inference(mixed_prompt: Prompt) -> None:
    segmenter: StubSegmenter = StubSegmenter(build_settings())
    with pytest.raises(ValueError, match=r"does not support \['visual'\] queries"):
        segmenter.segment(Image.new("RGB", (100, 50)), mixed_prompt)
    assert segmenter.mini_batch_sizes == []


def test_result_count_must_match_the_mini_batch() -> None:
    with pytest.raises(ValueError, match="expected 2 results"):
        TruncatingSegmenter(build_settings()).segment_images(
            [Image.new("RGB", (100, 50))] * 2, Prompt.from_class_names(("cat", "dog"))
        )


def test_settings_of_another_backend_are_rejected() -> None:
    settings: SegmenterSettings = SegmenterSettings(
        backend=SegmenterBackend.FLORENCE2,
        weights_path="stub",
        thresholds=SegmentationThresholds(confidence_threshold=0.0, nms_iou_threshold=None),
    )
    with pytest.raises(ValueError, match="needs sam3 settings"):
        StubSegmenter(settings)


def test_image_containers_are_accepted_as_rgb() -> None:
    segmenter: StubSegmenter = StubSegmenter(build_settings())
    gray = ImageContainer.register(np.zeros((50, 100), dtype=np.uint8), ChannelOrder.GRAY)
    bgr = ImageContainer.register(np.zeros((50, 100, 3), dtype=np.uint8), ChannelOrder.BGR)
    results: list[SegmentationResult] = segmenter.segment_images(
        [gray, bgr, Image.new("L", (100, 50))], Prompt.from_class_names(("cat", "dog"))
    )
    assert segmenter.received_modes == ["RGB", "RGB", "RGB"]
    assert [result.image_size for result in results] == [ImageSize(width=100, height=50)] * 3
