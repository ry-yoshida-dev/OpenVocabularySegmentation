from collections.abc import Sequence
from typing import ClassVar, cast

import torch
from PIL import Image
from transformers import BatchEncoding, EdgeTamModel, Sam2Model, Sam2Processor

from ...options import PromptableBackend
from ...promptable import ModelFrame, PromptableSegmenterSettings
from ..mask_decoder import DecodedMasks, MaskDecoderSegmenter, PreprocessedImages, PromptTensors


class Sam2PromptableSegmenter(MaskDecoderSegmenter):
    """
    SAM 2 / SAM 2.1 backed by ``transformers.Sam2Model``.

    SAM 2 stretches the image over its square input, and its image embedding is multi-scale: the decoder reads the
    high-resolution features next to the main embedding. The published checkpoints are video checkpoints
    (``sam2_video``) whose image part is loaded. Subclasses with the same processor and decoder interface load
    another model through ``_load_model``.
    """

    BACKEND: ClassVar[PromptableBackend] = PromptableBackend.SAM2

    def __init__(self, settings: PromptableSegmenterSettings) -> None:
        """
        Check the checkpoint, then load the processor and model.

        Parameters
        ----------
        settings : PromptableSegmenterSettings
            SAM 2 checkpoint, batching, device and precision.
        """
        super().__init__(settings)
        self._processor: Sam2Processor = Sam2Processor.from_pretrained(settings.weights_path)
        self._model: Sam2Model | EdgeTamModel = self._load_model(settings.weights_path)
        self._runtime.prepare_model(self._model)

    def _load_model(self, weights_path: str) -> Sam2Model | EdgeTamModel:
        """
        Load the image model of the checkpoint.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        Sam2Model | EdgeTamModel
            Model with SAM 2's image embedding and mask decoder interface.
        """
        return Sam2Model.from_pretrained(weights_path)

    def _preprocess(self, images: Sequence[Image.Image]) -> PreprocessedImages:
        inputs: BatchEncoding = self._processor(images=list(images), return_tensors="pt")
        return PreprocessedImages.stretched(inputs["pixel_values"], inputs["original_sizes"])

    def _embed(self, pixel_values: torch.Tensor) -> list[tuple[torch.Tensor, ...]]:
        embeddings: list[torch.Tensor] = self._model.get_image_embeddings(cast(torch.FloatTensor, pixel_values))
        return [
            tuple(embedding[index : index + 1] for embedding in embeddings) for index in range(pixel_values.shape[0])
        ]

    def _decode(self, features: tuple[torch.Tensor, ...], tensors: PromptTensors, is_multimask: bool) -> DecodedMasks:
        outputs = self._model(
            image_embeddings=list(features),
            input_points=tensors.points,
            input_labels=tensors.labels,
            input_boxes=tensors.boxes,
            input_masks=tensors.masks,
            multimask_output=is_multimask,
        )
        mask_logits: torch.Tensor = outputs.pred_masks
        iou_scores: torch.Tensor = outputs.iou_scores
        return DecodedMasks(low_resolution_logits=mask_logits[0].float().cpu(), iou_scores=iou_scores[0].float().cpu())

    def _upscale(self, low_resolution_logits: torch.Tensor, frame: ModelFrame) -> torch.Tensor:
        upscaled: list[torch.Tensor] = self._processor.post_process_masks(
            [low_resolution_logits[:, None]], [[frame.image_size.height, frame.image_size.width]], binarize=False
        )
        return upscaled[0][:, 0].float()
