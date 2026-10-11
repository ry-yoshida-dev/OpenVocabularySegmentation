from collections.abc import Sequence
from typing import ClassVar, cast

import torch
from PIL import Image
from transformers import BatchEncoding, Sam3TrackerModel, Sam3TrackerProcessor

from ...options import PromptableBackend
from ...promptable import ModelFrame, PromptableSegmenterSettings
from ..mask_decoder import DecodedMasks, MaskDecoderSegmenter, PreprocessedImages, PromptTensors


class Sam3PromptableSegmenter(MaskDecoderSegmenter):
    """
    Interactive (promptable visual) segmentation of SAM 3, backed by ``transformers.Sam3TrackerModel``.

    The SAM 3 tracker shares the SAM 3 vision backbone and works like SAM 2: the image is stretched over the square
    input, the embedding is multi-scale, and points, boxes and masks prompt one object each. It loads from the same
    checkpoint as SAM 3 concept segmentation (``facebook/sam3``, ``model_type`` ``sam3_video``).
    """

    BACKEND: ClassVar[PromptableBackend] = PromptableBackend.SAM3

    def __init__(self, settings: PromptableSegmenterSettings) -> None:
        """
        Check the checkpoint, then load the processor and model.

        Parameters
        ----------
        settings : PromptableSegmenterSettings
            SAM 3 checkpoint, batching, device and precision.
        """
        super().__init__(settings)
        self._processor: Sam3TrackerProcessor = Sam3TrackerProcessor.from_pretrained(settings.weights_path)
        self._model: Sam3TrackerModel = Sam3TrackerModel.from_pretrained(settings.weights_path)
        self._runtime.prepare_model(self._model)

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
