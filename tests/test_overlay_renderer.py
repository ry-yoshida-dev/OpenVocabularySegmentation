import numpy as np
import pytest
from image_container import ArrayImageContainer, ChannelOrder, ImageContainer
from PIL import Image

from open_vocabulary_segmentation import ImageSize, MaskOverlayRenderer, Prompt, SegmentationResult
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
    with pytest.raises(ValueError, match="palette channels"):
        MaskOverlayRenderer(palette=((0, 0, 300),))
