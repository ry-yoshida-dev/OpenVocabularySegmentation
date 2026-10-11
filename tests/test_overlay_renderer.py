import numpy as np
import pytest
from image_container import ArrayImageContainer, ChannelOrder, ImageContainer
from PIL import Image

from open_vocabulary_segmentation import (
    ConceptFrame,
    ImageSize,
    MaskGenerationResult,
    MaskOverlayRenderer,
    ObjectPrompt,
    ObjectSegmentationResult,
    Prompt,
    SegmentationResult,
    TrackedFrame,
)
from open_vocabulary_segmentation.array_types import BoolArray

IMAGE_SIZE: ImageSize = ImageSize(width=20, height=10)


def build_result() -> SegmentationResult:
    masks: BoolArray = np.zeros((2, IMAGE_SIZE.height, IMAGE_SIZE.width), dtype=np.bool_)
    masks[0, 0:6, 0:8] = True
    masks[1, 4:10, 6:14] = True
    return SegmentationResult.from_masks(
        masks=masks,
        confidences=np.array([0.4, 0.9]),
        query_ids=np.array([0, 1]),
        prompt=Prompt.from_class_names(("cat", "dog")),
        image_size=IMAGE_SIZE,
    )


def test_render_tints_instances_with_class_colors() -> None:
    renderer: MaskOverlayRenderer = MaskOverlayRenderer(
        opacity=1.0, contour_thickness=0, palette=((255, 0, 0), (0, 0, 255))
    )
    image = ImageContainer.register(Image.new("RGB", (20, 10), (10, 10, 10)), ChannelOrder.RGB)
    rendered: ArrayImageContainer = renderer.render(image, build_result())
    assert rendered.channel_order is ChannelOrder.RGB
    assert rendered.value[1, 1].tolist() == [255, 0, 0]
    assert rendered.value[5, 7].tolist() == [0, 0, 255]
    assert rendered.value[0, 19].tolist() == [10, 10, 10]


def test_render_blends_and_outlines() -> None:
    renderer: MaskOverlayRenderer = MaskOverlayRenderer(opacity=0.5, contour_thickness=1, palette=((200, 0, 0),))
    image: ArrayImageContainer = ArrayImageContainer(
        value=np.zeros((10, 20, 3), dtype=np.uint8), channel_order=ChannelOrder.RGB
    )
    rendered: ArrayImageContainer = renderer.render(image, build_result())
    assert rendered.value[2, 2].tolist() == [100, 0, 0]
    assert rendered.value[0, 0].tolist() == [200, 0, 0]
    assert renderer.color_of(3) == (200, 0, 0)


def test_invalid_renderer_settings_raise() -> None:
    with pytest.raises(ValueError, match="opacity"):
        MaskOverlayRenderer(opacity=1.5)
    with pytest.raises(ValueError, match="contour_thickness"):
        MaskOverlayRenderer(contour_thickness=-1)
    with pytest.raises(ValueError, match="palette must have"):
        MaskOverlayRenderer(palette=())
    with pytest.raises(ValueError, match="color channels"):
        MaskOverlayRenderer(palette=((0, 0, 300),))


def build_canvas() -> ArrayImageContainer:
    return ArrayImageContainer(value=np.zeros((10, 20, 3), dtype=np.uint8), channel_order=ChannelOrder.RGB)


def test_render_objects_draws_masks_and_prompts() -> None:
    renderer: MaskOverlayRenderer = MaskOverlayRenderer(
        opacity=1.0, contour_thickness=0, palette=((255, 0, 0), (0, 0, 255)), point_radius=1
    )
    masks: BoolArray = np.zeros((2, 10, 20), dtype=np.bool_)
    masks[0, 0:5, 0:5] = True
    masks[1, 5:10, 10:20] = True
    result: ObjectSegmentationResult = ObjectSegmentationResult(
        masks=masks,
        scores=np.array([0.9, 0.8]),
        stability_scores=np.array([0.9, 0.9]),
        prompt_indices=np.array([0, 1]),
        prompts=(
            ObjectPrompt.from_coordinates(positive_points=[(2, 2)], negative_points=[(8, 2)]),
            ObjectPrompt.from_coordinates(box_xyxy=(10, 5, 19, 9)),
        ),
        mask_logits=(None, None),
        image_size=IMAGE_SIZE,
    )
    plain: ArrayImageContainer = renderer.render_objects(build_canvas(), result, is_prompt_drawn=False)
    assert plain.value[4, 4].tolist() == [255, 0, 0]
    assert plain.value[7, 15].tolist() == [0, 0, 255]
    with_prompts: ArrayImageContainer = renderer.render_objects(build_canvas(), result)
    assert with_prompts.value[2, 2].tolist() == list(renderer.positive_point_color)
    assert with_prompts.value[2, 8].tolist() == list(renderer.negative_point_color)
    assert with_prompts.value[5, 12].tolist() == [0, 0, 255]


def test_render_tracked_concepts_and_generated() -> None:
    renderer: MaskOverlayRenderer = MaskOverlayRenderer(
        opacity=1.0, contour_thickness=0, palette=((255, 0, 0), (0, 255, 0))
    )
    masks: BoolArray = np.zeros((1, 10, 20), dtype=np.bool_)
    masks[0, 0:3, 0:3] = True
    tracked: TrackedFrame = TrackedFrame(
        frame_index=0, object_ids=np.array([1]), masks=masks, presence_scores=np.array([1.0]), image_size=IMAGE_SIZE
    )
    assert renderer.render_tracked(build_canvas(), tracked).value[1, 1].tolist() == [0, 255, 0]
    concepts: ConceptFrame = ConceptFrame(
        frame_index=0, instances=build_result(), track_ids=np.array([2, 5]), tracker_scores=np.array([0.9, 0.8])
    )
    assert renderer.render_concepts(build_canvas(), concepts).value[1, 1].tolist() == [255, 0, 0]
    generated: MaskGenerationResult = MaskGenerationResult(
        masks=masks,
        scores=np.array([0.9]),
        stability_scores=np.array([0.9]),
        point_xy=np.zeros((1, 2)),
        crop_xyxy=np.zeros((1, 4)),
        image_size=IMAGE_SIZE,
    )
    assert renderer.render_generated(build_canvas(), generated).value[1, 1].tolist() == [255, 0, 0]
    with pytest.raises(ValueError, match="color_ids"):
        renderer.render_masks(build_canvas(), IMAGE_SIZE, masks, np.array([0, 1]), np.array([0]))
    with pytest.raises(ValueError, match="point_radius"):
        MaskOverlayRenderer(point_radius=0)
