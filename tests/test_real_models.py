import numpy as np
import pytest
from geometry import Box2DFormat, Boxes2D
from huggingface_hub import try_to_load_from_cache
from PIL import Image, ImageDraw

from open_vocabulary_segmentation import (
    AutomaticMaskGenerator,
    BoxExemplars,
    ConceptFrame,
    ConceptTrackerSettings,
    ConceptTrackingParameters,
    DetectionThresholds,
    DetectorBackend,
    DetectorSettings,
    Device,
    ImageExemplars,
    MaskGenerationOptions,
    MaskGenerationResult,
    ObjectPrompt,
    ObjectSegmentationResult,
    ObjectTrackerSettings,
    OpenVocabularySegmenter,
    Prompt,
    PromptableBackend,
    PromptableSegmenter,
    PromptableSegmenterSettings,
    PropagationDirection,
    SegmentationResult,
    SegmentationThresholds,
    SegmenterBackend,
    SegmenterSettings,
    SemanticSegmentationResult,
    TrackedFrame,
    TrackerBackend,
    TrackingSession,
)
from open_vocabulary_segmentation.array_types import BoolArray, IntArray
from open_vocabulary_segmentation.backends.sam3 import Sam3Segmenter

pytestmark = pytest.mark.slow

IMAGE_WIDTH: int = 320
IMAGE_HEIGHT: int = 240
CIRCLE_XYXY: tuple[int, int, int, int] = (90, 50, 230, 190)
GATED_SAM3_WEIGHTS: str = "facebook/sam3"
SAM3_LITE_TEXT_WEIGHTS: str = "vil-uob/sam3-litetext-s0"
EDGETAM_WEIGHTS: str = "yonigozlan/EdgeTAM-hf"
FLORENCE2_WEIGHTS: str = "florence-community/Florence-2-base"
MIN_IOU: float = 0.9
BOX_TOLERANCE: int = 10
FLORENCE2_BACKENDS: frozenset[SegmenterBackend] = frozenset(
    {SegmenterBackend.FLORENCE2, SegmenterBackend.FLORENCE2_REFERRING}
)


def skip_if_gated(weights_path: str) -> None:
    if weights_path == GATED_SAM3_WEIGHTS and not isinstance(
        try_to_load_from_cache(GATED_SAM3_WEIGHTS, "config.json"), str
    ):
        pytest.skip(f"{GATED_SAM3_WEIGHTS} is gated; download it once after `hf auth login`.")


def build_circle_image() -> Image.Image:
    image: Image.Image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
    ImageDraw.Draw(image).ellipse(CIRCLE_XYXY, fill=(220, 30, 30))
    return image


def circle_mask(center_x: float = 160.0, center_y: float = 120.0, radius: float = 70.0) -> BoolArray:
    rows, columns = np.mgrid[:IMAGE_HEIGHT, :IMAGE_WIDTH]
    mask: BoolArray = ((columns - center_x) / radius) ** 2 + ((rows - center_y) / radius) ** 2 <= 1.0
    return mask


def iou(first: BoolArray, second: BoolArray) -> float:
    return float((first & second).sum() / max(int((first | second).sum()), 1))


def build_moving_frames(frame_count: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for index in range(frame_count):
        frame: Image.Image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
        center_x: int = 80 + 20 * index
        ImageDraw.Draw(frame).ellipse((center_x - 30, 90, center_x + 30, 150), fill=(220, 30, 30))
        frames.append(frame)
    return frames


def assert_box_near(pixel_xyxy: IntArray, expected_xyxy: tuple[int, int, int, int]) -> None:
    assert np.abs(pixel_xyxy - np.array(expected_xyxy)).max() <= BOX_TOLERANCE


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


@pytest.mark.parametrize(
    ("backend", "weights_path"),
    [
        (SegmenterBackend.SAM, "facebook/sam-vit-base"),
        (SegmenterBackend.SAM2, "facebook/sam2.1-hiera-tiny"),
        (SegmenterBackend.SAM3, GATED_SAM3_WEIGHTS),
        (SegmenterBackend.SAM3, SAM3_LITE_TEXT_WEIGHTS),
        (SegmenterBackend.EDGETAM, EDGETAM_WEIGHTS),
        (SegmenterBackend.FLORENCE2, FLORENCE2_WEIGHTS),
        (SegmenterBackend.FLORENCE2_REFERRING, FLORENCE2_WEIGHTS),
    ],
)
def test_real_model_segments_a_circle(backend: SegmenterBackend, weights_path: str) -> None:
    skip_if_gated(weights_path)
    segmenter: OpenVocabularySegmenter = build_settings(backend, weights_path).build()
    images: list[Image.Image] = [build_circle_image(), Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))]
    results: list[SegmentationResult] = segmenter.segment_images(images, Prompt.from_class_names(("circle",)))
    assert len(results) == len(images)
    for result in results:
        assert result.masks.shape[1:] == (IMAGE_HEIGHT, IMAGE_WIDTH)
        assert bool(np.all((result.confidences >= 0.0) & (result.confidences <= 1.0)))
    assert len(results[0]) >= 1
    assert_box_near(results[0].pixel_xyxy[0], CIRCLE_XYXY)
    assert max(iou(mask, circle_mask()) for mask in results[0].masks) >= MIN_IOU
    if not backend.is_box_prompted and backend not in FLORENCE2_BACKENDS:
        assert len(results[1]) == 0


@pytest.mark.parametrize(
    ("backend", "weights_path"),
    [
        (PromptableBackend.SAM, "facebook/sam-vit-base"),
        (PromptableBackend.SAM2, "facebook/sam2.1-hiera-tiny"),
        (PromptableBackend.SAM3, GATED_SAM3_WEIGHTS),
        (PromptableBackend.EDGETAM, EDGETAM_WEIGHTS),
    ],
)
def test_promptable_model_segments_points_boxes_and_refinements(backend: PromptableBackend, weights_path: str) -> None:
    skip_if_gated(weights_path)
    segmenter: PromptableSegmenter = PromptableSegmenterSettings(
        backend=backend, weights_path=weights_path, device=Device.CPU
    ).build()
    encoding = segmenter.encode(build_circle_image())
    result: ObjectSegmentationResult = segmenter.segment(
        encoding,
        [
            ObjectPrompt.from_coordinates(positive_points=[(160, 120)]),
            ObjectPrompt.from_coordinates(box_xyxy=(88, 48, 232, 192)),
            ObjectPrompt.from_coordinates(positive_points=[(160, 120), (120, 120)], negative_points=[(20, 20)]),
        ],
    )
    assert len(result) == 3
    for mask in result.masks:
        assert iou(mask, circle_mask()) >= MIN_IOU
    refined: ObjectSegmentationResult = segmenter.segment(encoding, [result[0].refine()])
    assert iou(refined.masks[0], circle_mask()) >= MIN_IOU


def test_florence2_segments_boxes() -> None:
    segmenter: PromptableSegmenter = PromptableSegmenterSettings(
        backend=PromptableBackend.FLORENCE2, weights_path=FLORENCE2_WEIGHTS, device=Device.CPU
    ).build()
    result: ObjectSegmentationResult = segmenter.segment(
        build_circle_image(), [ObjectPrompt.from_coordinates(box_xyxy=(85, 45, 235, 195))]
    )
    assert iou(result.masks[0], circle_mask()) >= MIN_IOU
    assert result.mask_logits == (None,)


def test_automatic_mask_generation_finds_the_circle() -> None:
    segmenter: PromptableSegmenter = PromptableSegmenterSettings(
        backend=PromptableBackend.SAM2, weights_path="facebook/sam2.1-hiera-tiny", device=Device.CPU
    ).build()
    result: MaskGenerationResult = AutomaticMaskGenerator(
        segmenter, MaskGenerationOptions(points_per_side=8, predicted_iou_threshold=0.8)
    ).generate(build_circle_image())
    assert max(iou(mask, circle_mask()) for mask in result.masks) >= MIN_IOU


@pytest.mark.parametrize(
    ("backend", "weights_path"),
    [
        (TrackerBackend.SAM2, "facebook/sam2.1-hiera-tiny"),
        (TrackerBackend.SAM3, GATED_SAM3_WEIGHTS),
        (TrackerBackend.EDGETAM, EDGETAM_WEIGHTS),
    ],
)
def test_object_tracker_follows_a_moving_circle(backend: TrackerBackend, weights_path: str) -> None:
    skip_if_gated(weights_path)
    tracker = ObjectTrackerSettings(backend=backend, weights_path=weights_path, device=Device.CPU).build()
    frames: list[Image.Image] = build_moving_frames(4)
    session: TrackingSession = tracker.start_session(frames)
    session.prompt_object(1, 7, ObjectPrompt.from_coordinates(positive_points=[(100, 120)]))
    tracked: dict[int, TrackedFrame] = {
        frame.frame_index: frame for frame in session.propagate(PropagationDirection.BIDIRECTIONAL)
    }
    assert sorted(tracked) == [0, 1, 2, 3]
    for frame_index, frame in tracked.items():
        assert iou(frame.object(7).mask, circle_mask(80 + 20 * frame_index, 120, 30)) >= MIN_IOU
    corrected: TrackedFrame = session.prompt_object(
        2, 7, ObjectPrompt.from_coordinates(positive_points=[(120, 120)], negative_points=[(140, 120)])
    )
    repropagated: dict[int, TrackedFrame] = {frame.frame_index: frame for frame in session.propagate()}
    assert np.array_equal(repropagated[2].object(7).mask, corrected.object(7).mask)
    backward: list[int] = [
        frame.frame_index for frame in session.propagate(PropagationDirection.BACKWARD, start_frame_index=2)
    ]
    assert backward == [2, 1, 0]
    stream: TrackingSession = tracker.start_stream()
    stream.track_next(frames[0], {1: ObjectPrompt.from_coordinates(box_xyxy=(48, 88, 112, 152))})
    last: TrackedFrame = stream.track_next(frames[1])
    assert iou(last.object(1).mask, circle_mask(100, 120, 30)) >= MIN_IOU


def test_concept_tracker_keeps_identities() -> None:
    skip_if_gated(GATED_SAM3_WEIGHTS)
    tracker = ConceptTrackerSettings(
        weights_path=GATED_SAM3_WEIGHTS,
        confidence_threshold=0.3,
        parameters=ConceptTrackingParameters(hotstart_delay=0),
        device=Device.CPU,
    ).build()
    session = tracker.start_session(build_moving_frames(3), Prompt.from_class_names(("red circle",)))
    frames: list[ConceptFrame] = list(session.propagate())
    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    track_ids: set[int] = {int(frame.track_ids[0]) for frame in frames}
    assert len(track_ids) == 1
    assert iou(frames[2].instances.masks[0], circle_mask(120, 120, 30)) >= MIN_IOU
    assert all(bool(np.all((frame.tracker_scores >= 0.0) & (frame.tracker_scores <= 1.0))) for frame in frames)
    backward: list[int] = [
        frame.frame_index for frame in session.propagate(PropagationDirection.BACKWARD, start_frame_index=1)
    ]
    assert backward == [1, 0]
    removed_tracks: list[int] = []
    for frame in session.propagate():
        assert not any(track_id in frame for track_id in removed_tracks)
        if not removed_tracks and len(frame):
            removed_tracks.append(int(frame.track_ids[0]))
            session.remove_track(removed_tracks[0])
    assert removed_tracks


def test_sam3_exemplars_and_semantics() -> None:
    skip_if_gated(GATED_SAM3_WEIGHTS)
    segmenter: OpenVocabularySegmenter = build_settings(SegmenterBackend.SAM3, GATED_SAM3_WEIGHTS).build()
    assert isinstance(segmenter, Sam3Segmenter)
    image: Image.Image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
    drawing: ImageDraw.ImageDraw = ImageDraw.Draw(image)
    for center_x in (60, 160, 260):
        drawing.ellipse((center_x - 30, 30, center_x + 30, 90), fill=(220, 30, 30))
    prompt: Prompt = Prompt.from_class_names(("red circle",))
    assert len(segmenter.segment(image, prompt)) == 3
    negative_boxes: Boxes2D = Boxes2D.register(
        value=np.array([[230.0, 30.0, 290.0, 90.0]]), box2d_format=Box2DFormat.XYXY
    )
    positive_boxes: Boxes2D = Boxes2D.register(
        value=np.array([[30.0, 30.0, 90.0, 90.0]]), box2d_format=Box2DFormat.XYXY
    )
    excluded: SegmentationResult = segmenter.segment(
        image,
        prompt,
        ImageExemplars({"red circle": BoxExemplars(positive_boxes=positive_boxes, negative_boxes=negative_boxes)}),
    )
    assert len(excluded) == 2
    assert int(excluded.pixel_xyxy[:, 0].max()) < 200
    visual_only: SegmentationResult = segmenter.segment(
        image,
        Prompt.from_class_names(("thing",)),
        ImageExemplars({"thing": BoxExemplars(positive_boxes=positive_boxes, is_text_ignored=True)}),
    )
    assert len(visual_only) == 3
    semantic: SemanticSegmentationResult = segmenter.segment_semantic(image, prompt)
    assert float(semantic.probability_of("red circle")[60, 160]) > 0.5
    assert float(semantic.probability_of("red circle")[200, 160]) < 0.5
