import numpy as np
import pytest
from huggingface_hub import try_to_load_from_cache
from PIL import Image, ImageDraw

from open_vocabulary_segmentation import (
    DetectionThresholds,
    DetectorBackend,
    DetectorSettings,
    Device,
    OpenVocabularySegmenter,
    Prompt,
    SegmentationResult,
    SegmentationThresholds,
    SegmenterBackend,
    SegmenterSettings,
)

pytestmark = pytest.mark.slow

IMAGE_WIDTH: int = 320
IMAGE_HEIGHT: int = 240
CIRCLE_XYXY: tuple[int, int, int, int] = (90, 50, 230, 190)
GATED_SAM3_WEIGHTS: str = "facebook/sam3"


def build_circle_image() -> Image.Image:
    image: Image.Image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
    ImageDraw.Draw(image).ellipse(CIRCLE_XYXY, fill=(220, 30, 30))
    return image


def build_grounding_dino_settings() -> DetectorSettings:
    return DetectorSettings(
        backend=DetectorBackend.GROUNDING_DINO,
        weights_path="IDEA-Research/grounding-dino-tiny",
        thresholds=DetectionThresholds(confidence_threshold=0.3, nms_iou_threshold=0.5),
        device=Device.CPU,
    )


def build_settings(backend: SegmenterBackend, weights_path: str) -> SegmenterSettings:
    return SegmenterSettings(
        backend=backend,
        weights_path=weights_path,
        thresholds=SegmentationThresholds(confidence_threshold=0.3, nms_iou_threshold=0.5),
        box_detector=build_grounding_dino_settings() if backend.is_box_prompted else None,
        batch_size=2,
        device=Device.CPU,
    )


def assert_valid_results(results: list[SegmentationResult], image_count: int) -> None:
    assert len(results) == image_count
    for result in results:
        assert result.masks.shape[1:] == (IMAGE_HEIGHT, IMAGE_WIDTH)
        assert bool(np.all((result.confidences >= 0.0) & (result.confidences <= 1.0)))
        assert set(result.class_names) <= {"circle"}


@pytest.mark.parametrize(
    ("backend", "weights_path"),
    [
        (SegmenterBackend.SAM, "facebook/sam-vit-base"),
        (SegmenterBackend.SAM2, "facebook/sam2.1-hiera-tiny"),
        (SegmenterBackend.SAM3, GATED_SAM3_WEIGHTS),
        (SegmenterBackend.FLORENCE2, "florence-community/Florence-2-base"),
    ],
)
def test_real_model_segments_a_circle(backend: SegmenterBackend, weights_path: str) -> None:
    if weights_path == GATED_SAM3_WEIGHTS and not isinstance(
        try_to_load_from_cache(GATED_SAM3_WEIGHTS, "config.json"), str
    ):
        pytest.skip(f"{GATED_SAM3_WEIGHTS} is gated; download it once after `hf auth login`.")
    segmenter: OpenVocabularySegmenter = build_settings(backend, weights_path).build()
    images: list[Image.Image] = [build_circle_image(), Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))]
    results: list[SegmentationResult] = segmenter.segment_images(images, Prompt.from_class_names(("circle",)))
    assert_valid_results(results, len(images))
    circle_result: SegmentationResult = results[0]
    assert len(circle_result) >= 1
    x_min, y_min, x_max, y_max = circle_result.pixel_xyxy[0].tolist()
    assert abs(x_min - CIRCLE_XYXY[0]) <= 10 and abs(y_min - CIRCLE_XYXY[1]) <= 10
    assert abs(x_max - CIRCLE_XYXY[2]) <= 10 and abs(y_max - CIRCLE_XYXY[3]) <= 10
