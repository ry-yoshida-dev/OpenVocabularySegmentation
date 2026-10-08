import numpy as np
import pytest
from image_container import ArrayImageContainer, ChannelOrder, ImageContainer, PILImageContainer
from PIL import Image

from open_vocabulary_segmentation import ImageSize, Prompt, Segmentation, SegmentationResult
from open_vocabulary_segmentation.array_types import BoolArray, FloatArray
from open_vocabulary_segmentation.result import MaskContour

IMAGE_SIZE: ImageSize = ImageSize(width=12, height=10)


def build_segmentation() -> Segmentation:
    mask: BoolArray = np.zeros((IMAGE_SIZE.height, IMAGE_SIZE.width), dtype=np.bool_)
    mask[2:6, 3:9] = True
    mask[2, 3] = False
    result: SegmentationResult = SegmentationResult.from_masks(
        masks=mask[None],
        confidences=np.array([0.8]),
        query_ids=np.array([0]),
        prompt=Prompt.from_class_names(("cup",)),
        image_size=IMAGE_SIZE,
    )
    return next(iter(result))


def build_image() -> ArrayImageContainer:
    pixels = np.arange(IMAGE_SIZE.height * IMAGE_SIZE.width * 3, dtype=np.int64).reshape(10, 12, 3) % 251
    return ArrayImageContainer(value=pixels.astype(np.uint8), channel_order=ChannelOrder.RGB)


def test_box_and_crop_slice_cover_the_mask() -> None:
    segmentation: Segmentation = build_segmentation()
    assert segmentation.box.value.tolist() == [3.0, 2.0, 9.0, 6.0]
    assert segmentation.crop_slice == (slice(2, 6), slice(3, 9))
    assert segmentation.area == 23
    assert segmentation.image_size == IMAGE_SIZE
    assert segmentation.binary_image.sum == 23


def test_crop_keeps_the_container_type() -> None:
    segmentation: Segmentation = build_segmentation()
    array_crop = segmentation.crop(build_image())
    assert isinstance(array_crop, ArrayImageContainer)
    assert array_crop.size == (6, 4)
    pil_image = ImageContainer.register(Image.new("RGB", (12, 10)), ChannelOrder.RGB)
    pil_crop = segmentation.crop(pil_image)
    assert isinstance(pil_crop, PILImageContainer)
    assert pil_crop.size == (6, 4)


def test_cutout_fills_pixels_outside_the_mask() -> None:
    segmentation: Segmentation = build_segmentation()
    image: ArrayImageContainer = build_image()
    cutout: ArrayImageContainer = segmentation.cutout(image, background_value=255)
    assert cutout.channel_order is ChannelOrder.RGB
    assert cutout.value.shape == (4, 6, 3)
    assert cutout.value[0, 0].tolist() == [255, 255, 255]
    assert cutout.value[1, 1].tolist() == image.value[3, 4].tolist()


def test_cutout_converts_the_channel_order_to_rgb() -> None:
    segmentation: Segmentation = build_segmentation()
    rgb: ArrayImageContainer = build_image()
    bgr: ArrayImageContainer = ArrayImageContainer(value=rgb.to_array(ChannelOrder.BGR), channel_order=ChannelOrder.BGR)
    assert segmentation.cutout(bgr).value.tolist() == segmentation.cutout(rgb).value.tolist()


def test_image_of_another_size_is_rejected() -> None:
    segmentation: Segmentation = build_segmentation()
    other: ArrayImageContainer = ArrayImageContainer(
        value=np.zeros((5, 5, 3), dtype=np.uint8), channel_order=ChannelOrder.RGB
    )
    with pytest.raises(ValueError, match="must be 12x10"):
        segmentation.crop(other)
    with pytest.raises(ValueError, match="background_value"):
        segmentation.cutout(build_image(), background_value=256)


def test_polygons_trace_every_region() -> None:
    mask: BoolArray = np.zeros((10, 12), dtype=np.bool_)
    mask[1:4, 1:4] = True
    mask[6:9, 7:11] = True
    mask[0, 11] = True
    polygons: list[FloatArray] = MaskContour.polygons(mask)
    assert len(polygons) == 2
    bounds = sorted((polygon.min(axis=0).tolist(), polygon.max(axis=0).tolist()) for polygon in polygons)
    assert bounds == [([1.0, 1.0], [3.0, 3.0]), ([7.0, 6.0], [10.0, 8.0])]
    assert len(build_segmentation().to_polygons()) == 1
