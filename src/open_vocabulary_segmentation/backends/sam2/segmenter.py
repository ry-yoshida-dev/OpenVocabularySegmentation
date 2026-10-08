from typing import ClassVar, cast

import torch
from PIL import Image
from transformers import BatchEncoding, Sam2Model, Sam2Processor

from ...array_types import BoolArray, FloatArray
from ...options import SegmenterBackend
from ...settings import SegmenterSettings
from ..box_prompted import BoxPromptedSegmenter


class Sam2Segmenter(BoxPromptedSegmenter[list[torch.Tensor]]):
    """
    SAM 2 / SAM 2.1 segmenter backed by ``transformers.Sam2Model``.

    Masks the boxes of the box detector with one mask per box (``multimask_output=False``); the multi-scale image
    embeddings of each image are computed once and shared by all its boxes. The published checkpoints are video
    checkpoints (``sam2_video``) whose image part is loaded.
    """

    BACKEND: ClassVar[SegmenterBackend] = SegmenterBackend.SAM2
    MODEL_TYPES: ClassVar[frozenset[str]] = frozenset({"sam2", "sam2_video"})

    def __init__(self, settings: SegmenterSettings) -> None:
        """
        Build the box detector, then load the processor and model.

        Parameters
        ----------
        settings : SegmenterSettings
            SAM 2 checkpoint, box detector, batching, device and thresholds.
        """
        super().__init__(settings)
        self._processor: Sam2Processor = Sam2Processor.from_pretrained(settings.weights_path)
        self._model: Sam2Model = Sam2Model.from_pretrained(settings.weights_path)
        self._runtime.prepare_model(self._model)

    def _preprocess(self, image: Image.Image, xyxy: FloatArray) -> BatchEncoding:
        inputs: BatchEncoding = self._processor(images=image, input_boxes=[xyxy.tolist()], return_tensors="pt")
        return inputs

    def _embed_images(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        return self._model.get_image_embeddings(cast(torch.FloatTensor, pixel_values))

    def _decode_masks(
        self, embeddings: list[torch.Tensor], image_index: int, input_boxes: torch.Tensor
    ) -> torch.Tensor:
        outputs = self._model(
            image_embeddings=[embedding[image_index : image_index + 1] for embedding in embeddings],
            input_boxes=input_boxes,
            multimask_output=False,
        )
        mask_logits: torch.Tensor = outputs.pred_masks
        return mask_logits

    def _upscale_masks(self, mask_logits: torch.Tensor, inputs: BatchEncoding) -> BoolArray:
        masks: list[torch.Tensor] = self._processor.post_process_masks(mask_logits, inputs["original_sizes"])
        return masks[0][:, 0].numpy().astype(bool)
