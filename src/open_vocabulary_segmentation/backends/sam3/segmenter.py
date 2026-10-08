from collections.abc import Sequence
from typing import ClassVar, cast

import numpy as np
import torch
from open_vocabulary_detector import ImageSize, Prompt, PromptKind
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Sam3Model, Sam3Processor

from ...array_types import BoolArray, FloatArray, IntArray
from ...options import SegmenterBackend
from ...result import SegmentationResult
from ...runtime import TorchRuntime
from ...segmenter import OpenVocabularySegmenter
from ...settings import SegmenterSettings
from ..checkpoint_config import CheckpointConfig
from .prompt_encoding import PromptEncoding


class Sam3Segmenter(OpenVocabularySegmenter):
    """
    SAM 3 segmenter backed by ``transformers.Sam3Model`` (promptable concept segmentation).

    SAM 3 finds and masks every instance of a text phrase on its own. The images of a mini-batch are encoded once,
    the text queries of a prompt are encoded once and reused while the same prompt is segmented, and the detection
    and mask decoders then run once per query over the mini-batch. An instance scores the product of its query
    match and the presence of the phrase in the image. Queries run independently, so one object may be found by
    several queries; the class-wise NMS merges those of one class.

    ``Sam3Model.get_text_features`` returns the output of its CLIP text encoder with the projected features attached
    as ``pooler_output`` (``PROJECTED_TEXT_FEATURES_ATTRIBUTE``), which is not part of that output type, so the
    attribute is read dynamically.
    """

    BACKEND: ClassVar[SegmenterBackend] = SegmenterBackend.SAM3
    MODEL_TYPES: ClassVar[frozenset[str]] = frozenset({"sam3", "sam3_video"})
    MASK_THRESHOLD: ClassVar[float] = 0.5
    PROJECTED_TEXT_FEATURES_ATTRIBUTE: ClassVar[str] = "pooler_output"

    def __init__(self, settings: SegmenterSettings) -> None:
        """
        Check the checkpoint, then load the processor and model.

        Parameters
        ----------
        settings : SegmenterSettings
            SAM 3 checkpoint, batching, device and thresholds.

        Raises
        ------
        ValueError
            If the checkpoint is not one of ``MODEL_TYPES``.
        """
        super().__init__(settings)
        CheckpointConfig.require_model_type(settings.weights_path, self.MODEL_TYPES)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        self._processor: Sam3Processor = Sam3Processor.from_pretrained(settings.weights_path)
        self._model: Sam3Model = Sam3Model.from_pretrained(settings.weights_path)
        self._runtime.prepare_model(self._model)
        self._active_encoding: PromptEncoding | None = None

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

    def _encoding_of(self, prompt: Prompt) -> PromptEncoding:
        if self._active_encoding is None or not self._active_encoding.is_for(prompt):
            self._active_encoding = self._encode(prompt)
        return self._active_encoding

    def _encode(self, prompt: Prompt) -> PromptEncoding:
        text_inputs: BatchEncoding = self._processor(text=list(prompt.query_texts), return_tensors="pt")
        input_ids: torch.Tensor = text_inputs["input_ids"].to(self._runtime.device)
        attention_mask: torch.Tensor = text_inputs["attention_mask"].to(self._runtime.device)
        with torch.inference_mode():
            text_outputs: object = cast(
                object,
                self._model.get_text_features(
                    input_ids=cast(torch.LongTensor, input_ids), attention_mask=attention_mask
                ),
            )
        features: object = getattr(text_outputs, self.PROJECTED_TEXT_FEATURES_ATTRIBUTE, None)
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"unexpected text encoder output: {type(text_outputs)}")
        return PromptEncoding(prompt=prompt, features=features, attention_mask=attention_mask)

    def _segment_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[SegmentationResult]:
        encoding: PromptEncoding = self._encoding_of(prompt)
        image_inputs: BatchFeature = self._processor.image_processor(images=list(images), return_tensors="pt")
        batch_size: int = len(images)
        image_sizes: list[ImageSize] = [ImageSize.from_image(image) for image in images]
        target_sizes: list[tuple[int, int]] = [(image_size.height, image_size.width) for image_size in image_sizes]
        image_masks: list[list[BoolArray]] = [[] for _ in images]
        image_confidences: list[list[FloatArray]] = [[] for _ in images]
        image_query_ids: list[list[IntArray]] = [[] for _ in images]
        with torch.inference_mode():
            vision_embeddings = self._model.get_vision_features(
                pixel_values=cast(torch.FloatTensor, self._runtime.to_model_input(image_inputs["pixel_values"]))
            )
            for query_id in range(len(prompt.queries)):
                outputs = self._model(
                    vision_embeds=vision_embeddings,
                    text_embeds=encoding.text_embeddings(query_id, batch_size),
                    attention_mask=encoding.query_attention_mask(query_id, batch_size),
                )
                instances: list[dict[str, torch.Tensor]] = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.thresholds.confidence_threshold,
                    mask_threshold=self.MASK_THRESHOLD,
                    target_sizes=target_sizes,
                )
                for image_index, image_instances in enumerate(instances):
                    confidences: FloatArray = image_instances["scores"].float().cpu().numpy().astype(np.float64)
                    image_masks[image_index].append(image_instances["masks"].bool().cpu().numpy())
                    image_confidences[image_index].append(confidences)
                    image_query_ids[image_index].append(np.full(len(confidences), query_id, dtype=np.int64))
        return [
            SegmentationResult.from_masks(
                masks=np.concatenate(masks),
                confidences=np.concatenate(confidences),
                query_ids=np.concatenate(query_ids),
                prompt=prompt,
                image_size=image_size,
            )
            for masks, confidences, query_ids, image_size in zip(
                image_masks, image_confidences, image_query_ids, image_sizes, strict=True
            )
        ]
