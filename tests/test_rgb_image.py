import numpy as np
import numpy.typing as npt
from image_container import ChannelOrder, ImageContainer
from PIL import Image

from open_vocabulary_segmentation.array_types import FloatArray, IntArray
from open_vocabulary_segmentation.image import RgbImage


def test_rgb_images_are_returned_unchanged() -> None:
    image: Image.Image = Image.new("RGB", (4, 2), (10, 20, 30))
    assert RgbImage.from_input(image) is image


def test_containers_are_converted_from_their_channel_order() -> None:
    bgr: npt.NDArray[np.uint8] = np.zeros((2, 4, 3), dtype=np.uint8)
    bgr[..., 0] = 255
    converted: Image.Image = RgbImage.from_input(ImageContainer.register(bgr, ChannelOrder.BGR))
    assert np.asarray(converted)[0, 0].tolist() == [0, 0, 255]


def test_high_bit_depth_grayscale_is_stretched_instead_of_clipped() -> None:
    depth: IntArray = np.array([[1000, 11000, 41000]], dtype=np.uint16)
    converted: Image.Image = RgbImage.from_input(Image.fromarray(depth))
    assert converted.mode == "RGB"
    assert np.asarray(converted)[0, :, 0].tolist() == [0, 64, 255]
    values: FloatArray = np.array([[np.nan, 2.0, 2.0]], dtype=np.float32)
    assert np.asarray(RgbImage.from_input(Image.fromarray(values)))[0, :, 0].tolist() == [0, 0, 0]
