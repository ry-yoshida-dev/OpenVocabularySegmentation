from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import torch
from open_vocabulary_detector import ImageSize
from PIL import Image

from ...array_types import BoolArray, FloatArray
from ...object_prompt import ObjectPrompt
from ...options import MaskOutput, PromptableBackend
from ...promptable import EncodedImage, ModelFrame, PromptableSegmenter, PromptableSegmenterSettings
from ...result import ObjectSegmentationResult
from ...runtime import TorchRuntime
from .detected_region import DetectedRegion
from .engine import Florence2Engine


class Florence2PromptableSegmenter(PromptableSegmenter):
    """
    Florence-2 region segmentation of boxes (``<REGION_TO_SEGMENTATION>``).

    Encoding an image only preprocesses it, since generation reads the pixels; each box is a beam search writing the
    polygon of the object in it, ``settings.decode_batch_size`` boxes per generation. Florence-2 gives no score and no
    logits: every mask has the score and stability ``GENERATED_SCORE``, no mask logits, and ``MaskOutput`` has no
    effect.
    """

    BACKEND: ClassVar[PromptableBackend] = PromptableBackend.FLORENCE2
    GENERATED_SCORE: ClassVar[float] = 1.0

    def __init__(self, settings: PromptableSegmenterSettings) -> None:
        """
        Load the Florence-2 engine.

        Parameters
        ----------
        settings : PromptableSegmenterSettings
            Florence-2 checkpoint, batching, device and precision.

        Raises
        ------
        ValueError
            If the checkpoint is not a Florence-2 checkpoint.
        """
        super().__init__(settings)
        self._engine: Florence2Engine = Florence2Engine(
            settings.weights_path, TorchRuntime(settings.device, settings.precision)
        )

    def _encode_mini_batch(self, images: Sequence[Image.Image]) -> list[EncodedImage]:
        pixel_values: torch.Tensor = self._engine.preprocess(images)
        input_size: int = int(pixel_values.shape[-1])
        return [
            EncodedImage(
                features=(pixel_values[index : index + 1],),
                frame=ModelFrame(
                    image_size=ImageSize.from_image(image),
                    resized_size=ImageSize(width=input_size, height=input_size),
                    input_size=input_size,
                ),
                backend=self.BACKEND,
                weights_path=self.settings.weights_path,
            )
            for index, image in enumerate(images)
        ]

    def _segment_encoded(
        self, encoding: EncodedImage, prompts: Sequence[ObjectPrompt], mask_output: MaskOutput
    ) -> ObjectSegmentationResult:
        image_size: ImageSize = encoding.image_size
        if not prompts:
            return ObjectSegmentationResult.empty(prompts, image_size)
        scale: FloatArray = np.array(
            [image_size.width, image_size.height, image_size.width, image_size.height], dtype=np.float64
        )
        regions: list[DetectedRegion] = [
            DetectedRegion(image_index=0, query_id=index, normalized_xyxy=self._xyxy_of(prompt) / scale)
            for index, prompt in enumerate(prompts)
        ]
        masks: list[BoolArray] = self._engine.segment_regions(
            encoding.features[0], regions, [image_size], self.settings.decode_batch_size
        )
        return ObjectSegmentationResult(
            masks=np.stack(masks),
            scores=np.full(len(prompts), self.GENERATED_SCORE, dtype=np.float64),
            stability_scores=np.full(len(prompts), self.GENERATED_SCORE, dtype=np.float64),
            prompt_indices=np.arange(len(prompts), dtype=np.int64),
            prompts=tuple(prompts),
            mask_logits=tuple(None for _ in prompts),
            image_size=image_size,
        )

    @staticmethod
    def _xyxy_of(prompt: ObjectPrompt) -> FloatArray:
        xyxy: FloatArray | None = prompt.xyxy
        if xyxy is None:
            raise ValueError("Florence-2 region segmentation needs a box in every prompt.")
        return xyxy
