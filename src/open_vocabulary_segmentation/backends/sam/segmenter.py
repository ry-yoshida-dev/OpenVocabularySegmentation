from typing import ClassVar

import torch
from PIL import Image
from transformers import BatchEncoding, SamModel, SamProcessor

from ...array_types import BoolArray, FloatArray
from ...options import SegmenterBackend
from ...settings import SegmenterSettings
from ..box_prompted import BoxPromptedSegmenter


class SamSegmenter(BoxPromptedSegmenter[torch.Tensor]):
    """
    Segment Anything segmenter backed by ``transformers.SamModel``.

    Masks the boxes of the box detector with one mask per box (``multimask_output=False``); the image embedding of
    each image is computed once and shared by all its boxes.
    """

    BACKEND: ClassVar[SegmenterBackend] = SegmenterBackend.SAM
    MODEL_TYPES: ClassVar[frozenset[str]] = frozenset({"sam"})

    def __init__(self, settings: SegmenterSettings) -> None:
        """
        Build the box detector, then load the processor and model.

        Parameters
        ----------
        settings : SegmenterSettings
            SAM checkpoint, box detector, batching, device and thresholds.
        """
        super().__init__(settings)
        self._processor: SamProcessor = SamProcessor.from_pretrained(settings.weights_path)
        self._model: SamModel = SamModel.from_pretrained(settings.weights_path)
        self._runtime.prepare_model(self._model)

    def _preprocess(self, image: Image.Image, xyxy: FloatArray) -> BatchEncoding:
        inputs: BatchEncoding = self._processor(images=image, input_boxes=[xyxy.tolist()], return_tensors="pt")
        return inputs

    def _embed_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings: torch.Tensor = self._model.get_image_embeddings(pixel_values)
        return embeddings

    def _decode_masks(self, embeddings: torch.Tensor, image_index: int, input_boxes: torch.Tensor) -> torch.Tensor:
        outputs = self._model(
            image_embeddings=embeddings[image_index : image_index + 1],
            input_boxes=input_boxes,
            multimask_output=False,
        )
        mask_logits: torch.Tensor = outputs.pred_masks
        return mask_logits

    def _upscale_masks(self, mask_logits: torch.Tensor, inputs: BatchEncoding) -> BoolArray:
        masks: list[torch.Tensor] = self._processor.post_process_masks(
            mask_logits, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )
        return masks[0][:, 0].numpy().astype(bool)
