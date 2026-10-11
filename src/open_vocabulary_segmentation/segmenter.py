from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from open_vocabulary_detector import ImageSize, Prompt, PromptKind
from PIL import Image

from .exemplars import ImageExemplars
from .image import ImageInput, RgbImage
from .options import SegmentationThresholds, SegmenterBackend
from .result import SegmentationResult
from .settings import SegmenterSettings


class OpenVocabularySegmenter(ABC):
    """
    Common base of every open-vocabulary segmenter, independent of the inference runtime.

    Subclasses declare the ``BACKENDS`` they implement and the query kinds they accept; a prompt using another kind
    is rejected before any computation. Segmenters supporting box exemplars (SAM 3) also take one
    ``ImageExemplars`` per image; others reject exemplars. The base owns the settings, mini-batching and the
    post-processing shared by every backend: confidence threshold, optional class-wise mask NMS, sort by
    confidence. Subclasses return the raw instances of a mini-batch, so post-processing behaves identically across
    backends.

    Instances found by different queries of one class (``"suv"``, ``"taxi"``) are merged by the class-wise NMS into
    the most confident one, which still records the query that matched.

    Attributes
    ----------
    settings : SegmenterSettings
        Backend, weights, batching, device and thresholds.
    """

    BACKENDS: ClassVar[frozenset[SegmenterBackend]]

    def __init__(self, settings: SegmenterSettings) -> None:
        """
        Parameters
        ----------
        settings : SegmenterSettings
            Backend, weights, batching, device and thresholds.

        Raises
        ------
        ValueError
            If the settings are for another backend.
        """
        if settings.backend not in self.BACKENDS:
            raise ValueError(
                f"{type(self).__name__} needs settings of {sorted(backend.value for backend in self.BACKENDS)}. "
                + f"got {settings.backend}"
            )
        self.settings: SegmenterSettings = settings

    @property
    @abstractmethod
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        """
        Query kinds the segmenter accepts.

        Returns
        -------
        frozenset[PromptKind]
            Accepted kinds; for box-prompted backends, those of the box detector.
        """

    @property
    def is_exemplar_supported(self) -> bool:
        """
        Whether the segmenter reads box exemplars next to the prompt.

        Returns
        -------
        bool
            False unless a subclass supports exemplars.
        """
        return False

    @property
    def batch_size(self) -> int:
        """
        Number of images processed per forward pass.

        Returns
        -------
        int
            Mini-batch size used by ``segment_images``.
        """
        return self.settings.batch_size

    @property
    def thresholds(self) -> SegmentationThresholds:
        """
        Post-processing thresholds in use.

        Returns
        -------
        SegmentationThresholds
            Configured thresholds.
        """
        return self.settings.thresholds

    @abstractmethod
    def _segment_mini_batch(
        self, images: Sequence[Image.Image], prompt: Prompt, exemplars: Sequence[ImageExemplars | None]
    ) -> list[SegmentationResult]:
        """
        Segment at most ``batch_size`` RGB images.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images.
        prompt : Prompt
            Classes to segment.
        exemplars : Sequence[ImageExemplars | None]
            Validated exemplars of each image; all ``None`` unless ``is_exemplar_supported``.

        Returns
        -------
        list[SegmentationResult]
            Raw instances of each image in input order, before the shared post-processing.
        """

    def segment(self, image: ImageInput, prompt: Prompt, exemplars: ImageExemplars | None = None) -> SegmentationResult:
        """
        Segment the prompt classes in a single image.

        Parameters
        ----------
        image : ImageInput
            Input image, a PIL image or an ``image_container`` container of any channel order.
        prompt : Prompt
            Classes to segment.
        exemplars : ImageExemplars | None, optional
            Box exemplars drawn on ``image``, for segmenters supporting them.

        Returns
        -------
        SegmentationResult
            Instances of the image.
        """
        return self.segment_images([image], prompt, None if exemplars is None else [exemplars])[0]

    def segment_images(
        self,
        images: Sequence[ImageInput],
        prompt: Prompt,
        exemplars: Sequence[ImageExemplars | None] | None = None,
    ) -> list[SegmentationResult]:
        """
        Segment the prompt classes in many images, processed in mini-batches.

        Parameters
        ----------
        images : Sequence[ImageInput]
            Input images of arbitrary size, as PIL images of any mode or ``image_container`` containers of any channel
            order.
        prompt : Prompt
            Classes to segment.
        exemplars : Sequence[ImageExemplars | None] | None, optional
            Box exemplars of each image (``None`` for an image without), for segmenters supporting them.

        Raises
        ------
        ValueError
            If ``images`` is empty, the prompt uses an unsupported query kind, exemplars are given to a segmenter
            without exemplar support or do not match the images, or a backend returns another number of results than
            images.
        KeyError
            If exemplars refer to a class that is not in the prompt.

        Returns
        -------
        list[SegmentationResult]
            One result per image, in input order, ordered by descending confidence.
        """
        if not images:
            raise ValueError("images must contain at least one image.")
        self._validate_prompt(prompt)
        image_exemplars: list[ImageExemplars | None] = self._exemplars_per_image(images, exemplars)
        results: list[SegmentationResult] = []
        for start in range(0, len(images), self.batch_size):
            rgb_images: list[Image.Image] = [
                RgbImage.from_input(image) for image in images[start : start + self.batch_size]
            ]
            mini_batch_exemplars: list[ImageExemplars | None] = image_exemplars[start : start + self.batch_size]
            for rgb_image, exemplars_of_image in zip(rgb_images, mini_batch_exemplars, strict=True):
                if exemplars_of_image is not None:
                    exemplars_of_image.validate_for(prompt, ImageSize.from_image(rgb_image))
            raw_results: list[SegmentationResult] = self._segment_mini_batch(rgb_images, prompt, mini_batch_exemplars)
            if len(raw_results) != len(rgb_images):
                raise ValueError(f"expected {len(rgb_images)} results. got {len(raw_results)}")
            results.extend(self._postprocess(raw_result) for raw_result in raw_results)
        return results

    def _validate_prompt(self, prompt: Prompt) -> None:
        unsupported_kinds: frozenset[PromptKind] = prompt.kinds - self.supported_prompt_kinds
        if unsupported_kinds:
            raise ValueError(
                f"{type(self).__name__} does not support {sorted(kind.value for kind in unsupported_kinds)} queries. "
                + f"supported: {sorted(kind.value for kind in self.supported_prompt_kinds)}"
            )

    def _exemplars_per_image(
        self, images: Sequence[ImageInput], exemplars: Sequence[ImageExemplars | None] | None
    ) -> list[ImageExemplars | None]:
        if exemplars is None:
            return [None] * len(images)
        if len(exemplars) != len(images):
            raise ValueError(f"expected exemplars for {len(images)} images. got {len(exemplars)}")
        if not self.is_exemplar_supported and any(image_exemplars is not None for image_exemplars in exemplars):
            raise ValueError(f"{type(self).__name__} does not support box exemplars.")
        return list(exemplars)

    def _postprocess(self, result: SegmentationResult) -> SegmentationResult:
        """
        Apply the confidence threshold, the optional class-wise mask NMS and sort by confidence.

        Parameters
        ----------
        result : SegmentationResult
            Raw instances of one image.

        Returns
        -------
        SegmentationResult
            Kept instances ordered by descending confidence.
        """
        thresholded: SegmentationResult = result.filter_by_confidence(self.thresholds.confidence_threshold)
        nms_iou_threshold: float | None = self.thresholds.nms_iou_threshold
        if nms_iou_threshold is None:
            return thresholded.sort_by_confidence()
        return thresholded.non_maximum_suppression(nms_iou_threshold)
