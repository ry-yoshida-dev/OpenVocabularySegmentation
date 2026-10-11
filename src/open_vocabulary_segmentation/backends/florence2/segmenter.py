from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import torch
from open_vocabulary_detector import ImageSize, Prompt, PromptKind
from PIL import Image

from ...array_types import BoolArray
from ...exemplars import ImageExemplars
from ...options import SegmenterBackend
from ...result import SegmentationResult
from ...runtime import TorchRuntime
from ...segmenter import OpenVocabularySegmenter
from ...settings import SegmenterSettings
from .detected_region import DetectedRegion
from .engine import Florence2Engine


class Florence2Segmenter(OpenVocabularySegmenter):
    """
    Florence-2 open-vocabulary instance segmenter.

    Instances are found in two steps with one model: ``<OPEN_VOCABULARY_DETECTION>`` writes the boxes of every
    instance of a text query (one generation per query over the mini-batch), then ``<REGION_TO_SEGMENTATION>``
    writes the polygon of each box. Images are preprocessed once per mini-batch and shared by both steps. The model
    gives no score, so every instance has the confidence ``SEGMENTATION_CONFIDENCE``; a box whose polygon encloses
    no pixel is dropped.
    """

    BACKENDS: ClassVar[frozenset[SegmenterBackend]] = frozenset({SegmenterBackend.FLORENCE2})
    SEGMENTATION_CONFIDENCE: ClassVar[float] = 1.0

    def __init__(self, settings: SegmenterSettings) -> None:
        """
        Load the Florence-2 engine.

        Parameters
        ----------
        settings : SegmenterSettings
            Florence-2 checkpoint, batching, device, precision and thresholds.

        Raises
        ------
        ValueError
            If the checkpoint is not a Florence-2 checkpoint.
        """
        super().__init__(settings)
        self._engine: Florence2Engine = Florence2Engine(
            settings.weights_path, TorchRuntime(settings.device, settings.precision)
        )

    @property
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        """
        Query kinds the segmenter accepts.

        Returns
        -------
        frozenset[PromptKind]
            ``TEXT`` only.
        """
        return frozenset({PromptKind.TEXT})

    def _segment_mini_batch(
        self, images: Sequence[Image.Image], prompt: Prompt, exemplars: Sequence[ImageExemplars | None]
    ) -> list[SegmentationResult]:
        pixel_values: torch.Tensor = self._engine.preprocess(images)
        image_sizes: list[ImageSize] = [ImageSize.from_image(image) for image in images]
        regions: list[DetectedRegion] = self._engine.detect_regions(pixel_values, prompt.query_texts)
        region_masks: list[BoolArray] = self._engine.segment_regions(pixel_values, regions, image_sizes)
        results: list[SegmentationResult] = []
        for image_index, image_size in enumerate(image_sizes):
            region_indices: list[int] = [
                index for index, region in enumerate(regions) if region.image_index == image_index
            ]
            results.append(
                SegmentationResult.from_masks(
                    masks=np.stack([region_masks[index] for index in region_indices])
                    if region_indices
                    else np.zeros((0, image_size.height, image_size.width), dtype=np.bool_),
                    confidences=np.full(len(region_indices), self.SEGMENTATION_CONFIDENCE, dtype=np.float64),
                    query_ids=np.array([regions[index].query_id for index in region_indices], dtype=np.int64),
                    prompt=prompt,
                    image_size=image_size,
                )
            )
        return results
