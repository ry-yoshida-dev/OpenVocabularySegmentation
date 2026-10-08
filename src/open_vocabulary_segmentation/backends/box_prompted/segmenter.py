from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import torch
from open_vocabulary_detector import (
    DetectionResult,
    DetectorSettings,
    ImageSize,
    OpenVocabularyDetector,
    Prompt,
    PromptKind,
)
from PIL import Image
from transformers import BatchEncoding

from ...array_types import BoolArray, FloatArray
from ...result import SegmentationResult
from ...runtime import TorchRuntime
from ...segmenter import OpenVocabularySegmenter
from ...settings import SegmenterSettings
from ..checkpoint_config import CheckpointConfig


class BoxPromptedSegmenter[EmbeddingT](OpenVocabularySegmenter):
    """
    Base of segmenters that mask the boxes found by an open-vocabulary detector (Grounded-SAM style).

    The box detector from ``SegmenterSettings.box_detector`` finds the instances of the prompt, so the accepted query
    kinds, the confidences and the matched queries are those of the detector; its thresholds and NMS select the
    boxes. Each image with at least one box is preprocessed once together with its boxes, the images of a
    mini-batch are embedded in one pass, and the boxes of each image are decoded in chunks of ``BOX_CHUNK_SIZE``
    into one mask per box. Images without boxes skip the segmentation model.

    Subclasses provide the model-specific steps through ``_preprocess``, ``_embed_images``, ``_decode_masks`` and
    ``_upscale_masks``; ``EmbeddingT`` is the image embedding type of the model.
    """

    MODEL_TYPES: ClassVar[frozenset[str]]
    BOX_CHUNK_SIZE: ClassVar[int] = 32

    def __init__(self, settings: SegmenterSettings) -> None:
        """
        Check the checkpoint and build the box detector.

        Parameters
        ----------
        settings : SegmenterSettings
            Checkpoint, box detector, batching, device and thresholds.

        Raises
        ------
        ValueError
            If the settings have no box detector or the checkpoint is not one of ``MODEL_TYPES``.
        """
        super().__init__(settings)
        box_detector_settings: DetectorSettings | None = settings.box_detector
        if box_detector_settings is None:
            raise ValueError(f"{type(self).__name__} needs box_detector settings.")
        CheckpointConfig.require_model_type(settings.weights_path, self.MODEL_TYPES)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        self._box_detector: OpenVocabularyDetector = box_detector_settings.build()

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
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        """
        Query kinds the box detector accepts.

        Returns
        -------
        frozenset[PromptKind]
            Kinds supported by the box detector.
        """
        return self._box_detector.supported_prompt_kinds

    @abstractmethod
    def _preprocess(self, image: Image.Image, xyxy: FloatArray) -> BatchEncoding:
        """
        Preprocess one image together with its boxes.

        Parameters
        ----------
        image : Image.Image
            RGB image.
        xyxy : FloatArray
            Absolute XYXY pixel boxes, shape (N, 4) with N > 0.

        Returns
        -------
        BatchEncoding
            Processor output with ``pixel_values`` of shape (1, C, H, W) and ``input_boxes`` of shape (1, N, 4).
        """

    @abstractmethod
    def _embed_images(self, pixel_values: torch.Tensor) -> EmbeddingT:
        """
        Run the image encoder once over a mini-batch.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Preprocessed images on the runtime device and dtype, shape (B, C, H, W).

        Returns
        -------
        EmbeddingT
            Image embeddings of the mini-batch.
        """

    @abstractmethod
    def _decode_masks(self, embeddings: EmbeddingT, image_index: int, input_boxes: torch.Tensor) -> torch.Tensor:
        """
        Decode one mask per box of one image.

        Parameters
        ----------
        embeddings : EmbeddingT
            Image embeddings of the mini-batch.
        image_index : int
            Index of the image in ``embeddings``.
        input_boxes : torch.Tensor
            Boxes in model input coordinates on the runtime device and dtype, shape (1, N, 4).

        Returns
        -------
        torch.Tensor
            Low-resolution mask logits, shape (1, N, 1, h, w).
        """

    @abstractmethod
    def _upscale_masks(self, mask_logits: torch.Tensor, inputs: BatchEncoding) -> BoolArray:
        """
        Resize low-resolution mask logits to the source image and binarize them.

        Parameters
        ----------
        mask_logits : torch.Tensor
            Low-resolution mask logits as float32 on the CPU, shape (1, N, 1, h, w).
        inputs : BatchEncoding
            Output of ``_preprocess`` for the image.

        Returns
        -------
        BoolArray
            Masks in the source image, shape (N, H, W).
        """

    def _segment_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[SegmentationResult]:
        detections: list[DetectionResult] = self._box_detector.detect_images(images, prompt)
        masks: list[BoolArray] = self._mask_boxes(images, [detection.xyxy for detection in detections])
        return [
            SegmentationResult.from_masks(
                masks=image_masks,
                confidences=detection.confidences,
                query_ids=detection.query_ids,
                prompt=prompt,
                image_size=ImageSize.from_image(image),
            )
            for image_masks, detection, image in zip(masks, detections, images, strict=True)
        ]

    def _mask_boxes(self, images: Sequence[Image.Image], boxes: Sequence[FloatArray]) -> list[BoolArray]:
        masks: list[BoolArray] = [np.zeros((0, image.height, image.width), dtype=np.bool_) for image in images]
        prompted_indices: list[int] = [index for index, xyxy in enumerate(boxes) if len(xyxy)]
        if not prompted_indices:
            return masks
        inputs: list[BatchEncoding] = [self._preprocess(images[index], boxes[index]) for index in prompted_indices]
        pixel_values: torch.Tensor = torch.cat([image_inputs["pixel_values"] for image_inputs in inputs])
        with torch.inference_mode():
            embeddings: EmbeddingT = self._embed_images(self._runtime.to_model_input(pixel_values))
            for batch_index, (image_index, image_inputs) in enumerate(zip(prompted_indices, inputs, strict=True)):
                input_boxes: torch.Tensor = self._runtime.to_model_input(image_inputs["input_boxes"])
                masks[image_index] = np.concatenate(
                    [
                        self._upscale_masks(
                            self._decode_masks(
                                embeddings, batch_index, input_boxes[:, start : start + self.BOX_CHUNK_SIZE]
                            )
                            .float()
                            .cpu(),
                            image_inputs,
                        )
                        for start in range(0, input_boxes.shape[1], self.BOX_CHUNK_SIZE)
                    ]
                )
        return masks
