from collections.abc import Sequence
from typing import ClassVar, cast

import numpy as np
import torch
from open_vocabulary_detector import ImageSize, Prompt, PromptKind
from open_vocabulary_detector.backends.florence2 import LocationParser, TokenGenerator
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Florence2ForConditionalGeneration, Florence2Processor

from ...array_types import BoolArray
from ...options import SegmenterBackend
from ...result import SegmentationResult
from ...runtime import TorchRuntime
from ...segmenter import OpenVocabularySegmenter
from ...settings import SegmenterSettings
from ..checkpoint_config import CheckpointConfig
from .detected_region import DetectedRegion
from .location_tokens import LocationTokens
from .polygon_parser import PolygonParser
from .polygon_rasterizer import PolygonRasterizer


class Florence2Segmenter(OpenVocabularySegmenter):
    """
    Florence-2 segmenter backed by ``transformers.Florence2ForConditionalGeneration``.

    Florence-2 is generative and segments one region per generation, so instances are found in two steps with one
    model: ``<OPEN_VOCABULARY_DETECTION>`` writes the boxes of every instance of a text query (one generation per
    query over the mini-batch), then ``<REGION_TO_SEGMENTATION>`` writes the polygon of each box (generations
    batched by ``REGION_CHUNK_SIZE`` regions, independently of the image ``batch_size``). Images are preprocessed
    once per mini-batch and shared by both steps. The model gives no score, so every instance has the confidence
    ``SEGMENTATION_CONFIDENCE``; a box whose polygon encloses no pixel is dropped.
    """

    BACKEND: ClassVar[SegmenterBackend] = SegmenterBackend.FLORENCE2
    MODEL_TYPES: ClassVar[frozenset[str]] = frozenset({"florence2"})
    DETECTION_TASK_TOKEN: ClassVar[str] = "<OPEN_VOCABULARY_DETECTION>"
    SEGMENTATION_TASK_TOKEN: ClassVar[str] = "<REGION_TO_SEGMENTATION>"
    SEGMENTATION_CONFIDENCE: ClassVar[float] = 1.0
    MAX_NEW_TOKENS: ClassVar[int] = 1024
    BEAM_COUNT: ClassVar[int] = 3
    REGION_CHUNK_SIZE: ClassVar[int] = 8

    def __init__(self, settings: SegmenterSettings) -> None:
        """
        Check the checkpoint, then load the processor and model.

        Parameters
        ----------
        settings : SegmenterSettings
            Florence-2 checkpoint, batching, device and thresholds.

        Raises
        ------
        ValueError
            If the checkpoint is not one of ``MODEL_TYPES``.
        """
        super().__init__(settings)
        CheckpointConfig.require_model_type(settings.weights_path, self.MODEL_TYPES)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        self._processor: Florence2Processor = Florence2Processor.from_pretrained(settings.weights_path)
        self._model: Florence2ForConditionalGeneration = Florence2ForConditionalGeneration.from_pretrained(
            settings.weights_path
        )
        self._runtime.prepare_model(self._model)
        self._image_placeholder: str = self._read_image_placeholder()

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

    def _read_image_placeholder(self) -> str:
        image_token: object = cast(object, self._processor.image_token)
        image_token_count: object = cast(object, self._processor.num_image_tokens)
        if not isinstance(image_token, str) or not isinstance(image_token_count, int):
            raise TypeError(
                f"expected a str image token and an int token count. got {type(image_token)} "
                + f"and {type(image_token_count)}"
            )
        return image_token * image_token_count

    def _segment_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[SegmentationResult]:
        image_inputs: BatchFeature = self._processor.image_processor(images=list(images), return_tensors="pt")
        pixel_values: torch.Tensor = self._runtime.to_model_input(image_inputs["pixel_values"])
        image_sizes: list[ImageSize] = [ImageSize.from_image(image) for image in images]
        regions: list[DetectedRegion] = self._detect_regions(pixel_values, prompt)
        region_masks: list[BoolArray] = self._segment_regions(pixel_values, regions, image_sizes)
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

    def _detect_regions(self, pixel_values: torch.Tensor, prompt: Prompt) -> list[DetectedRegion]:
        regions: list[DetectedRegion] = []
        image_count: int = int(pixel_values.shape[0])
        for query_id, query_text in enumerate(prompt.query_texts):
            generated_texts: list[str] = self._generate(
                pixel_values, [self._task_prompt(self.DETECTION_TASK_TOKEN, query_text.strip())] * image_count
            )
            for image_index, generated_text in enumerate(generated_texts):
                regions.extend(
                    DetectedRegion(image_index=image_index, query_id=query_id, normalized_xyxy=normalized_xyxy)
                    for normalized_xyxy in LocationParser.parse(generated_text)
                )
        return regions

    def _segment_regions(
        self, pixel_values: torch.Tensor, regions: Sequence[DetectedRegion], image_sizes: Sequence[ImageSize]
    ) -> list[BoolArray]:
        masks: list[BoolArray] = []
        for start in range(0, len(regions), self.REGION_CHUNK_SIZE):
            chunk: Sequence[DetectedRegion] = regions[start : start + self.REGION_CHUNK_SIZE]
            image_indices: torch.Tensor = torch.tensor(
                [region.image_index for region in chunk], dtype=torch.long, device=pixel_values.device
            )
            generated_texts: list[str] = self._generate(
                pixel_values.index_select(0, image_indices),
                [
                    self._task_prompt(self.SEGMENTATION_TASK_TOKEN, LocationTokens.encode_box(region.normalized_xyxy))
                    for region in chunk
                ],
            )
            masks.extend(
                PolygonRasterizer.rasterize(PolygonParser.parse(generated_text), image_sizes[region.image_index])
                for region, generated_text in zip(chunk, generated_texts, strict=True)
            )
        return masks

    def _task_prompt(self, task_token: str, task_input: str) -> str:
        return self._processor.task_prompts_with_input[task_token].format(input=task_input)

    def _generate(self, pixel_values: torch.Tensor, task_prompts: Sequence[str]) -> list[str]:
        prompt_texts: list[str] = [
            self._image_placeholder
            + self._processor.tokenizer.bos_token
            + task_prompt
            + self._processor.tokenizer.eos_token
            for task_prompt in task_prompts
        ]
        encoding: BatchEncoding = self._processor.tokenizer(
            prompt_texts, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        with torch.inference_mode():
            generated_ids: torch.Tensor = cast(TokenGenerator, self._model).generate(
                input_ids=encoding["input_ids"].to(self._runtime.device),
                attention_mask=encoding["attention_mask"].to(self._runtime.device),
                pixel_values=pixel_values,
                max_new_tokens=self.MAX_NEW_TOKENS,
                num_beams=self.BEAM_COUNT,
                do_sample=False,
            )
        generated_texts: list[str] = self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return generated_texts
