from image_container import ChannelOrder, ImageContainerType
from image_container.types import UInt8Image
from open_vocabulary_detector import ImageSize


class SourceImage:
    """
    Checks that an image is the one a segmentation result refers to before its pixels are combined with masks.
    """

    @staticmethod
    def validate(image: ImageContainerType, image_size: ImageSize) -> None:
        """
        Check that an image has the size of the masks.

        Parameters
        ----------
        image : ImageContainerType
            Source image of any container type and channel order.
        image_size : ImageSize
            Size of the masks.

        Raises
        ------
        ValueError
            If the image size differs from ``image_size``.
        """
        if (image.width, image.height) != (image_size.width, image_size.height):
            raise ValueError(
                f"image must be {image_size.width}x{image_size.height} like the masks. got {image.width}x{image.height}"
            )

    @classmethod
    def rgb_array(cls, image: ImageContainerType, image_size: ImageSize) -> UInt8Image:
        """
        Pixels of an image as an RGB array, after checking its size.

        Parameters
        ----------
        image : ImageContainerType
            Source image of any container type and channel order.
        image_size : ImageSize
            Size of the masks.

        Returns
        -------
        UInt8Image
            RGB pixels, shape (H, W, 3).
        """
        cls.validate(image, image_size)
        return image.to_array(ChannelOrder.RGB)
