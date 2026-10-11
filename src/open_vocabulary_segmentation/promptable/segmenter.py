from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import ClassVar

from PIL import Image

from ..image import ImageInput, RgbImage
from ..object_prompt import MaskLogits, ObjectPrompt
from ..options import MaskOutput, PromptableBackend, PromptElement
from ..result import ObjectSegmentationResult
from .encoded_image import EncodedImage
from .promptable_input import PromptableInput
from .settings import PromptableSegmenterSettings


class PromptableSegmenter(ABC):
    """
    Common base of every promptable segmenter: segments the single object each ``ObjectPrompt`` marks.

    An image is encoded once (``encode``), then prompted any number of times; ``segment`` also accepts raw images
    and encodes them on the fly. Prompts are validated against the image and the backend before any computation:
    unsupported prompt elements, coordinates off the image, binary masks of another size and mask logits of another
    checkpoint or image are rejected.

    Attributes
    ----------
    settings : PromptableSegmenterSettings
        Backend, weights, batching, device and precision.
    """

    BACKEND: ClassVar[PromptableBackend]

    def __init__(self, settings: PromptableSegmenterSettings) -> None:
        """
        Parameters
        ----------
        settings : PromptableSegmenterSettings
            Backend, weights, batching, device and precision.

        Raises
        ------
        ValueError
            If the settings are for another backend.
        """
        if settings.backend is not self.BACKEND:
            raise ValueError(f"{type(self).__name__} needs {self.BACKEND} settings. got {settings.backend}")
        self.settings: PromptableSegmenterSettings = settings

    @property
    def supported_prompt_elements(self) -> frozenset[PromptElement]:
        """
        Prompt elements the segmenter accepts.

        Returns
        -------
        frozenset[PromptElement]
            Elements of ``BACKEND``.
        """
        return self.BACKEND.supported_prompt_elements

    def encode(self, image: ImageInput) -> EncodedImage:
        """
        Encode one image for repeated prompting.

        Parameters
        ----------
        image : ImageInput
            PIL image of any mode or ``image_container`` container of any channel order.

        Returns
        -------
        EncodedImage
            Image embedding of this segmenter.
        """
        return self.encode_images([image])[0]

    def encode_images(self, images: Sequence[ImageInput]) -> list[EncodedImage]:
        """
        Encode many images, ``settings.batch_size`` per forward pass.

        Parameters
        ----------
        images : Sequence[ImageInput]
            Input images of arbitrary size.

        Returns
        -------
        list[EncodedImage]
            One encoding per image, in input order.
        """
        encodings: list[EncodedImage] = []
        for start in range(0, len(images), self.settings.batch_size):
            mini_batch: Sequence[ImageInput] = images[start : start + self.settings.batch_size]
            encodings.extend(self._encode_mini_batch([RgbImage.from_input(image) for image in mini_batch]))
        return encodings

    def segment(
        self,
        image: PromptableInput,
        prompts: Sequence[ObjectPrompt],
        mask_output: MaskOutput = MaskOutput.AUTO,
    ) -> ObjectSegmentationResult:
        """
        Segment the object of every prompt in one image.

        Parameters
        ----------
        image : PromptableInput
            Raw image, or an encoding of this segmenter to skip the image encoder.
        prompts : Sequence[ObjectPrompt]
            One prompt per object.
        mask_output : MaskOutput, optional
            Which predicted masks to return per prompt.

        Raises
        ------
        ValueError
            If a prompt is invalid for the image or the backend, or the encoding is from another segmenter.

        Returns
        -------
        ObjectSegmentationResult
            Masks in prompt order.
        """
        return self.segment_images([image], [prompts], mask_output)[0]

    def segment_images(
        self,
        images: Sequence[PromptableInput],
        prompts_per_image: Sequence[Sequence[ObjectPrompt]],
        mask_output: MaskOutput = MaskOutput.AUTO,
    ) -> list[ObjectSegmentationResult]:
        """
        Segment the prompted objects of many images; raw images are encoded ``settings.batch_size`` at a time.

        Parameters
        ----------
        images : Sequence[PromptableInput]
            Raw images or encodings of this segmenter.
        prompts_per_image : Sequence[Sequence[ObjectPrompt]]
            Prompts of each image, in image order.
        mask_output : MaskOutput, optional
            Which predicted masks to return per prompt.

        Raises
        ------
        ValueError
            If the numbers of images and prompt lists differ, a prompt is invalid for its image or the backend, or
            an encoding is from another segmenter.

        Returns
        -------
        list[ObjectSegmentationResult]
            One result per image, in input order.
        """
        if len(images) != len(prompts_per_image):
            raise ValueError(f"expected prompts for {len(images)} images. got {len(prompts_per_image)}")
        for prompts in prompts_per_image:
            self._validate_elements(prompts)
        results: list[ObjectSegmentationResult] = []
        for start in range(0, len(images), self.settings.batch_size):
            mini_batch: Sequence[PromptableInput] = images[start : start + self.settings.batch_size]
            encodings: list[EncodedImage] = self._encodings_of(mini_batch)
            for encoding, prompts in zip(
                encodings, prompts_per_image[start : start + self.settings.batch_size], strict=True
            ):
                self._validate_prompts(encoding, prompts)
                results.append(self._segment_encoded(encoding, prompts, mask_output))
        return results

    @abstractmethod
    def _encode_mini_batch(self, images: Sequence[Image.Image]) -> list[EncodedImage]:
        """
        Encode at most ``settings.batch_size`` RGB images.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images.

        Returns
        -------
        list[EncodedImage]
            One encoding per image, in input order.
        """

    @abstractmethod
    def _segment_encoded(
        self, encoding: EncodedImage, prompts: Sequence[ObjectPrompt], mask_output: MaskOutput
    ) -> ObjectSegmentationResult:
        """
        Segment validated prompts on an encoded image.

        Parameters
        ----------
        encoding : EncodedImage
            Encoding of this segmenter.
        prompts : Sequence[ObjectPrompt]
            Prompts valid for the image and the backend.
        mask_output : MaskOutput
            Which predicted masks to return per prompt.

        Returns
        -------
        ObjectSegmentationResult
            Masks in prompt order.
        """

    def _encodings_of(self, images: Sequence[PromptableInput]) -> list[EncodedImage]:
        for image in images:
            if isinstance(image, EncodedImage):
                self._validate_encoding(image)
        raw_images: list[Image.Image] = [
            RgbImage.from_input(image) for image in images if not isinstance(image, EncodedImage)
        ]
        encoded_raw: Iterator[EncodedImage] = iter(self._encode_mini_batch(raw_images) if raw_images else [])
        return [image if isinstance(image, EncodedImage) else next(encoded_raw) for image in images]

    def _validate_encoding(self, encoding: EncodedImage) -> None:
        if encoding.backend is not self.BACKEND or encoding.weights_path != self.settings.weights_path:
            raise ValueError(
                f"the image was encoded by {encoding.backend} {encoding.weights_path}; this segmenter runs "
                + f"{self.BACKEND} {self.settings.weights_path}."
            )

    def _validate_elements(self, prompts: Sequence[ObjectPrompt]) -> None:
        for prompt in prompts:
            unsupported_elements: frozenset[PromptElement] = prompt.elements - self.supported_prompt_elements
            if unsupported_elements:
                raise ValueError(
                    f"{type(self).__name__} does not support {sorted(element.value for element in unsupported_elements)} "
                    + f"prompts. supported: {sorted(element.value for element in self.supported_prompt_elements)}"
                )
            if prompt.elements == frozenset({PromptElement.MASK}) and not self.BACKEND.is_mask_only_prompt_supported:
                raise ValueError(
                    f"{self.BACKEND} cannot segment from a mask alone; add points or a box to the mask prompt."
                )

    def _validate_prompts(self, encoding: EncodedImage, prompts: Sequence[ObjectPrompt]) -> None:
        for prompt in prompts:
            prompt.validate_for(encoding.image_size)
            if isinstance(prompt.mask, MaskLogits) and not prompt.mask.is_compatible_with(
                self.BACKEND, self.settings.weights_path, encoding.image_size
            ):
                raise ValueError(
                    "mask logits must come from this segmenter and an image of the same size: got "
                    + f"{prompt.mask.backend} {prompt.mask.weights_path} for a "
                    + f"{prompt.mask.image_size.width}x{prompt.mask.image_size.height} image."
                )
