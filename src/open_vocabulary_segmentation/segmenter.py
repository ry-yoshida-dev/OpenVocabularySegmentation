from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import ClassVar

from image_container import ImageContainerType
from open_vocabulary_detector import Prompt, PromptKind
from PIL import Image

from .options import SegmentationThresholds, SegmenterBackend
from .result import SegmentationResult
from .settings import SegmenterSettings


class OpenVocabularySegmenter(ABC):
    """
    Common base of every open-vocabulary segmenter, independent of the inference runtime.

    Subclasses declare the ``BACKEND`` they implement and the query kinds they accept; a prompt using another kind
    is rejected before any computation. The base owns the settings, mini-batching and the post-processing shared by
    every backend: confidence threshold, optional class-wise mask NMS, sort by confidence. Subclasses return the raw
    instances of a mini-batch, so post-processing behaves identically across backends.

    Instances found by different queries of one class (``"suv"``, ``"taxi"``) are merged by the class-wise NMS into
    the most confident one, which still records the query that matched.

    Attributes
    ----------
    settings : SegmenterSettings
        Backend, weights, batching, device and thresholds.
    """

    BACKEND: ClassVar[SegmenterBackend]

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
        if settings.backend is not self.BACKEND:
            raise ValueError(f"{type(self).__name__} needs {self.BACKEND} settings. got {settings.backend}")
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
    def _segment_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[SegmentationResult]:
        """
        Segment at most ``batch_size`` RGB images.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images.
        prompt : Prompt
            Classes to segment.

        Returns
        -------
        list[SegmentationResult]
            Raw instances of each image in input order, before the shared post-processing.
        """

    def segment(self, image: Image.Image | ImageContainerType, prompt: Prompt) -> SegmentationResult:
        """
        Segment the prompt classes in a single image.

        Parameters
        ----------
        image : Image.Image | ImageContainerType
            Input image, a PIL image or an ``image_container`` container of any channel order.
        prompt : Prompt
            Classes to segment.

        Returns
        -------
        SegmentationResult
            Instances of the image.
        """
        return self.segment_images([image], prompt)[0]

    def segment_images(
        self, images: Sequence[Image.Image | ImageContainerType], prompt: Prompt
    ) -> list[SegmentationResult]:
        """
        Segment the prompt classes in many images, processed in mini-batches.

        Parameters
        ----------
        images : Sequence[Image.Image | ImageContainerType]
            Input images of arbitrary size, as PIL images of any mode or ``image_container`` containers of any channel
            order.
        prompt : Prompt
            Classes to segment.

        Raises
        ------
        ValueError
            If ``images`` is empty, the prompt uses an unsupported query kind, or a backend returns another number
            of results than images.

        Returns
        -------
        list[SegmentationResult]
            One result per image, in input order, ordered by descending confidence.
        """
        if not images:
            raise ValueError("images must contain at least one image.")
        self._validate_prompt(prompt)
        results: list[SegmentationResult] = []
        for mini_batch in self._mini_batches(images):
            raw_results: list[SegmentationResult] = self._segment_mini_batch(
                [self._to_rgb(image) for image in mini_batch], prompt
            )
            if len(raw_results) != len(mini_batch):
                raise ValueError(f"expected {len(mini_batch)} results. got {len(raw_results)}")
            results.extend(self._postprocess(raw_result) for raw_result in raw_results)
        return results

    def _validate_prompt(self, prompt: Prompt) -> None:
        unsupported_kinds: frozenset[PromptKind] = prompt.kinds - self.supported_prompt_kinds
        if unsupported_kinds:
            raise ValueError(
                f"{type(self).__name__} does not support {sorted(kind.value for kind in unsupported_kinds)} queries. "
                + f"supported: {sorted(kind.value for kind in self.supported_prompt_kinds)}"
            )

    def _mini_batches(
        self, images: Sequence[Image.Image | ImageContainerType]
    ) -> Iterator[Sequence[Image.Image | ImageContainerType]]:
        for start in range(0, len(images), self.batch_size):
            yield images[start : start + self.batch_size]

    @staticmethod
    def _to_rgb(image: Image.Image | ImageContainerType) -> Image.Image:
        pil_image: Image.Image = image if isinstance(image, Image.Image) else image.to_PIL()
        return pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")

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
