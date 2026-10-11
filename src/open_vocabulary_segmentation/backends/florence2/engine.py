from collections.abc import Sequence
from typing import ClassVar, cast

import torch
from open_vocabulary_detector import ImageSize
from open_vocabulary_detector.backends.florence2 import LocationParser, TokenGenerator
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Florence2ForConditionalGeneration, Florence2Processor

from ...array_types import BoolArray
from ...runtime import TorchRuntime
from ..checkpoint_config import CheckpointConfig
from .detected_region import DetectedRegion
from .location_tokens import LocationTokens
from .polygon_parser import PolygonParser
from .polygon_rasterizer import PolygonRasterizer


class Florence2Engine:
    """
    Florence-2 tasks shared by the Florence-2 segmenters, backed by ``transformers.Florence2ForConditionalGeneration``.

    Florence-2 is generative: every task is a beam search (``BEAM_COUNT`` beams, up to ``MAX_NEW_TOKENS`` tokens)
    writing boxes or polygons as location tokens. ``<OPEN_VOCABULARY_DETECTION>`` writes the boxes of every instance
    of a phrase, ``<REGION_TO_SEGMENTATION>`` the polygon of a box, and ``<REFERRING_EXPRESSION_SEGMENTATION>`` the
    polygons of the single object a phrase refers to. Region generations are batched by ``REGION_CHUNK_SIZE``.
    """

    MODEL_TYPES: ClassVar[frozenset[str]] = frozenset({"florence2"})
    DETECTION_TASK_TOKEN: ClassVar[str] = "<OPEN_VOCABULARY_DETECTION>"
    REGION_SEGMENTATION_TASK_TOKEN: ClassVar[str] = "<REGION_TO_SEGMENTATION>"
    REFERRING_SEGMENTATION_TASK_TOKEN: ClassVar[str] = "<REFERRING_EXPRESSION_SEGMENTATION>"
    MAX_NEW_TOKENS: ClassVar[int] = 1024
    BEAM_COUNT: ClassVar[int] = 3
    REGION_CHUNK_SIZE: ClassVar[int] = 8

    def __init__(self, weights_path: str, runtime: TorchRuntime) -> None:
        """
        Check the checkpoint, then load the processor and model.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.
        runtime : TorchRuntime
            Device and dtype to run the model with.

        Raises
        ------
        ValueError
            If the checkpoint is not one of ``MODEL_TYPES``.
        """
        CheckpointConfig.require_model_type(weights_path, self.MODEL_TYPES)
        self._runtime: TorchRuntime = runtime
        self._processor: Florence2Processor = Florence2Processor.from_pretrained(weights_path)
        self._model: Florence2ForConditionalGeneration = Florence2ForConditionalGeneration.from_pretrained(weights_path)
        self._runtime.prepare_model(self._model)
        self._image_placeholder: str = self._read_image_placeholder()

    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """
        Resize and normalize RGB images for the vision encoder.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images.

        Returns
        -------
        torch.Tensor
            Pixel values on the runtime device and dtype, shape (B, C, S, S).
        """
        image_inputs: BatchFeature = self._processor.image_processor(images=list(images), return_tensors="pt")
        return self._runtime.to_model_input(image_inputs["pixel_values"])

    def detect_regions(self, pixel_values: torch.Tensor, query_texts: Sequence[str]) -> list[DetectedRegion]:
        """
        Detect every instance of every phrase, one generation per phrase over the images.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Preprocessed images, shape (B, C, S, S).
        query_texts : Sequence[str]
            Phrases in query-id order.

        Returns
        -------
        list[DetectedRegion]
            Boxes with their image and query.
        """
        regions: list[DetectedRegion] = []
        image_count: int = int(pixel_values.shape[0])
        for query_id, query_text in enumerate(query_texts):
            generated_texts: list[str] = self._generate(
                pixel_values, [self._task_prompt(self.DETECTION_TASK_TOKEN, query_text.strip())] * image_count
            )
            for image_index, generated_text in enumerate(generated_texts):
                regions.extend(
                    DetectedRegion(image_index=image_index, query_id=query_id, normalized_xyxy=normalized_xyxy)
                    for normalized_xyxy in LocationParser.parse(generated_text)
                )
        return regions

    def segment_regions(
        self,
        pixel_values: torch.Tensor,
        regions: Sequence[DetectedRegion],
        image_sizes: Sequence[ImageSize],
        chunk_size: int | None = None,
    ) -> list[BoolArray]:
        """
        Write the polygon of every box and fill it into a mask.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Preprocessed images, shape (B, C, S, S).
        regions : Sequence[DetectedRegion]
            Boxes with their image.
        image_sizes : Sequence[ImageSize]
            Size of each image.
        chunk_size : int | None, optional
            Regions generated together; ``None`` uses ``REGION_CHUNK_SIZE``.

        Returns
        -------
        list[BoolArray]
            Mask of each region over its image, shape (H, W) each, in region order.
        """
        regions_per_chunk: int = self.REGION_CHUNK_SIZE if chunk_size is None else chunk_size
        masks: list[BoolArray] = []
        for start in range(0, len(regions), regions_per_chunk):
            chunk: Sequence[DetectedRegion] = regions[start : start + regions_per_chunk]
            image_indices: torch.Tensor = torch.tensor(
                [region.image_index for region in chunk], dtype=torch.long, device=pixel_values.device
            )
            generated_texts: list[str] = self._generate(
                pixel_values.index_select(0, image_indices),
                [
                    self._task_prompt(
                        self.REGION_SEGMENTATION_TASK_TOKEN, LocationTokens.encode_box(region.normalized_xyxy)
                    )
                    for region in chunk
                ],
            )
            masks.extend(
                PolygonRasterizer.rasterize(PolygonParser.parse(generated_text), image_sizes[region.image_index])
                for region, generated_text in zip(chunk, generated_texts, strict=True)
            )
        return masks

    def segment_referred(
        self, pixel_values: torch.Tensor, query_texts: Sequence[str], image_sizes: Sequence[ImageSize]
    ) -> list[list[BoolArray]]:
        """
        Segment the single object each phrase refers to, one generation per phrase over the images.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Preprocessed images, shape (B, C, S, S).
        query_texts : Sequence[str]
            Referring phrases in query-id order.
        image_sizes : Sequence[ImageSize]
            Size of each image.

        Returns
        -------
        list[list[BoolArray]]
            For each image, the mask of each phrase in query-id order, shape (H, W) each; empty when the model
            writes no polygon.
        """
        image_count: int = int(pixel_values.shape[0])
        masks: list[list[BoolArray]] = [[] for _ in range(image_count)]
        for query_text in query_texts:
            generated_texts: list[str] = self._generate(
                pixel_values,
                [self._task_prompt(self.REFERRING_SEGMENTATION_TASK_TOKEN, query_text.strip())] * image_count,
            )
            for image_index, generated_text in enumerate(generated_texts):
                masks[image_index].append(
                    PolygonRasterizer.rasterize(PolygonParser.parse(generated_text), image_sizes[image_index])
                )
        return masks

    def _read_image_placeholder(self) -> str:
        image_token: object = cast(object, self._processor.image_token)
        image_token_count: object = cast(object, self._processor.num_image_tokens)
        if not isinstance(image_token, str) or not isinstance(image_token_count, int):
            raise TypeError(
                f"expected a str image token and an int token count. got {type(image_token)} "
                + f"and {type(image_token_count)}"
            )
        return image_token * image_token_count

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
                input_ids=self._runtime.to_device(encoding["input_ids"]),
                attention_mask=self._runtime.to_device(encoding["attention_mask"]),
                pixel_values=pixel_values,
                max_new_tokens=self.MAX_NEW_TOKENS,
                num_beams=self.BEAM_COUNT,
                do_sample=False,
            )
        generated_texts: list[str] = self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return generated_texts
