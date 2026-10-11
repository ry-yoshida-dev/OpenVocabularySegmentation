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
from .engine import Florence2Engine


class Florence2ReferringSegmenter(OpenVocabularySegmenter):
    """
    Florence-2 referring expression segmenter: each text query names one object, e.g. ``"the dog on the left"``.

    ``<REFERRING_EXPRESSION_SEGMENTATION>`` writes the polygons of the referred object in a single generation per
    query over the mini-batch, so it is about twice as fast as ``Florence2Segmenter`` but returns at most one
    instance per query. The model gives no score: every instance has the confidence ``SEGMENTATION_CONFIDENCE``, and
    a query whose polygons enclose no pixel yields no instance.
    """

    BACKENDS: ClassVar[frozenset[SegmenterBackend]] = frozenset({SegmenterBackend.FLORENCE2_REFERRING})
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
        referred_masks: list[list[BoolArray]] = self._engine.segment_referred(
            pixel_values, prompt.query_texts, image_sizes
        )
        query_count: int = len(prompt.queries)
        return [
            SegmentationResult.from_masks(
                masks=np.stack(masks),
                confidences=np.full(query_count, self.SEGMENTATION_CONFIDENCE, dtype=np.float64),
                query_ids=np.arange(query_count, dtype=np.int64),
                prompt=prompt,
                image_size=image_size,
            )
            for masks, image_size in zip(referred_masks, image_sizes, strict=True)
        ]
