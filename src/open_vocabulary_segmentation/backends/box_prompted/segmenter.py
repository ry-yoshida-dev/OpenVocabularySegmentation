from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from geometry import Box2D, Box2DFormat
from open_vocabulary_detector import (
    DetectionResult,
    DetectorSettings,
    ImageSize,
    OpenVocabularyDetector,
    Prompt,
    PromptKind,
)
from PIL import Image

from ...array_types import BoolArray, FloatArray, IntArray
from ...exemplars import ImageExemplars
from ...object_prompt import ObjectPrompt
from ...options import MaskOutput, PromptableBackend, SegmenterBackend
from ...promptable import EncodedImage, PromptableSegmenter, PromptableSegmenterSettings
from ...result import ObjectSegmentationResult, SegmentationResult
from ...segmenter import OpenVocabularySegmenter
from ...settings import SegmenterSettings


class BoxPromptedSegmenter(OpenVocabularySegmenter):
    """
    Open-vocabulary segmenter masking the boxes of an open-vocabulary detector with a promptable model
    (Grounded-SAM style).

    The box detector from ``SegmenterSettings.box_detector`` finds the instances of the prompt, so the accepted query
    kinds, the confidences and the matched queries are those of the detector; its thresholds and NMS select the
    boxes. Boxes are clipped to the image, the images with at least one box are encoded together by the promptable
    segmenter of the backend, and every box becomes one mask of the single-mask head. Images without boxes skip the
    segmentation model.
    """

    BACKENDS: ClassVar[frozenset[SegmenterBackend]] = frozenset(
        backend for backend in SegmenterBackend if backend.is_box_prompted
    )

    def __init__(self, settings: SegmenterSettings) -> None:
        """
        Build the box detector and the promptable segmenter masking its boxes.

        Parameters
        ----------
        settings : SegmenterSettings
            Checkpoint, box detector, batching, device, precision and thresholds.

        Raises
        ------
        ValueError
            If the settings have no box detector or the checkpoint does not fit the backend.
        """
        super().__init__(settings)
        box_detector_settings: DetectorSettings | None = settings.box_detector
        promptable_backend: PromptableBackend | None = settings.backend.promptable_backend
        if box_detector_settings is None or promptable_backend is None:
            raise ValueError(f"{type(self).__name__} needs box_detector settings and a box-prompted backend.")
        self._box_detector: OpenVocabularyDetector = box_detector_settings.build()
        self._mask_segmenter: PromptableSegmenter = self._build_mask_segmenter(
            PromptableSegmenterSettings(
                backend=promptable_backend,
                weights_path=settings.weights_path,
                batch_size=settings.batch_size,
                device=settings.device,
                precision=settings.precision,
            )
        )

    @property
    def box_detector(self) -> OpenVocabularyDetector:
        """
        Detector finding the boxes to mask.

        Returns
        -------
        OpenVocabularyDetector
            Detector built from ``settings.box_detector``.
        """
        return self._box_detector

    @property
    def mask_segmenter(self) -> PromptableSegmenter:
        """
        Promptable segmenter masking the boxes, also usable for interactive prompts on the same weights.

        Returns
        -------
        PromptableSegmenter
            Segmenter of the backend's promptable family.
        """
        return self._mask_segmenter

    @property
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        """
        Query kinds the box detector accepts.

        Returns
        -------
        frozenset[PromptKind]
            Kinds supported by the box detector.
        """
        return self._box_detector.supported_prompt_kinds

    def _build_mask_segmenter(self, settings: PromptableSegmenterSettings) -> PromptableSegmenter:
        """
        Load the promptable segmenter masking the boxes.

        Parameters
        ----------
        settings : PromptableSegmenterSettings
            Settings derived from the segmenter settings.

        Returns
        -------
        PromptableSegmenter
            Loaded segmenter.
        """
        return settings.build()

    def _segment_mini_batch(
        self, images: Sequence[Image.Image], prompt: Prompt, exemplars: Sequence[ImageExemplars | None]
    ) -> list[SegmentationResult]:
        detections: list[DetectionResult] = self._box_detector.detect_images(images, prompt)
        prompted_indices: list[int] = [index for index, detection in enumerate(detections) if len(detection)]
        encodings: dict[int, EncodedImage] = dict(
            zip(
                prompted_indices,
                self._mask_segmenter.encode_images([images[index] for index in prompted_indices]),
                strict=True,
            )
        )
        results: list[SegmentationResult] = []
        for index, (image, detection) in enumerate(zip(images, detections, strict=True)):
            image_size: ImageSize = ImageSize.from_image(image)
            encoding: EncodedImage | None = encodings.get(index)
            masks: BoolArray = (
                np.zeros((0, image_size.height, image_size.width), dtype=np.bool_)
                if encoding is None
                else self._mask_boxes(encoding, detection.xyxy)
            )
            results.append(
                SegmentationResult.from_masks(
                    masks=masks,
                    confidences=detection.confidences,
                    query_ids=detection.query_ids,
                    prompt=prompt,
                    image_size=image_size,
                )
            )
        return results

    def _mask_boxes(self, encoding: EncodedImage, xyxy: FloatArray) -> BoolArray:
        image_size: ImageSize = encoding.image_size
        clipped: FloatArray = np.clip(
            xyxy, 0.0, np.array([image_size.width, image_size.height, image_size.width, image_size.height])
        )
        is_valid: BoolArray = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
        masks: BoolArray = np.zeros((len(clipped), image_size.height, image_size.width), dtype=np.bool_)
        valid_indices: IntArray = np.flatnonzero(is_valid).astype(np.int64)
        if not len(valid_indices):
            return masks
        prompts: list[ObjectPrompt] = [
            ObjectPrompt(box=Box2D.register(value=clipped[index], box2d_format=Box2DFormat.XYXY))
            for index in valid_indices.tolist()
        ]
        result: ObjectSegmentationResult = self._mask_segmenter.segment(encoding, prompts, MaskOutput.SINGLE)
        masks[valid_indices] = result.masks
        return masks
