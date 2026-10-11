from collections.abc import Sequence
from typing import ClassVar

import torch
from open_vocabulary_detector import ImageSize
from PIL import Image
from transformers import BatchEncoding, SamModel, SamProcessor

from ...options import PromptableBackend
from ...promptable import ModelFrame, PromptableSegmenterSettings
from ..mask_decoder import DecodedMasks, MaskDecoderSegmenter, PreprocessedImages, PromptTensors


class SamPromptableSegmenter(MaskDecoderSegmenter):
    """
    Segment Anything backed by ``transformers.SamModel``.

    SAM resizes the longest image side to the model input size and pads the rest, so prompts are scaled by the
    resize factor and masks are cropped back from the padded frame. The image embedding is a single tensor.
    """

    BACKEND: ClassVar[PromptableBackend] = PromptableBackend.SAM

    def __init__(self, settings: PromptableSegmenterSettings) -> None:
        """
        Check the checkpoint, then load the processor and model.

        Parameters
        ----------
        settings : PromptableSegmenterSettings
            SAM checkpoint, batching, device and precision.
        """
        super().__init__(settings)
        self._processor: SamProcessor = SamProcessor.from_pretrained(settings.weights_path)
        self._model: SamModel = SamModel.from_pretrained(settings.weights_path)
        self._runtime.prepare_model(self._model)

    def _preprocess(self, images: Sequence[Image.Image]) -> PreprocessedImages:
        inputs: BatchEncoding = self._processor(images=list(images), return_tensors="pt")
        pixel_values: torch.Tensor = inputs["pixel_values"]
        original_sizes: list[list[int]] = inputs["original_sizes"].tolist()
        reshaped_sizes: list[list[int]] = inputs["reshaped_input_sizes"].tolist()
        return PreprocessedImages(
            pixel_values=pixel_values,
            frames=tuple(
                ModelFrame(
                    image_size=ImageSize(width=original[1], height=original[0]),
                    resized_size=ImageSize(width=reshaped[1], height=reshaped[0]),
                    input_size=int(pixel_values.shape[-1]),
                )
                for original, reshaped in zip(original_sizes, reshaped_sizes, strict=True)
            ),
        )

    def _embed(self, pixel_values: torch.Tensor) -> list[tuple[torch.Tensor, ...]]:
        embeddings: torch.Tensor = self._model.get_image_embeddings(pixel_values)
        return [(embeddings[index : index + 1],) for index in range(embeddings.shape[0])]

    def _decode(self, features: tuple[torch.Tensor, ...], tensors: PromptTensors, is_multimask: bool) -> DecodedMasks:
        outputs = self._model(
            image_embeddings=features[0],
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
            [low_resolution_logits[:, None]],
            [[frame.image_size.height, frame.image_size.width]],
            [[frame.resized_size.height, frame.resized_size.width]],
            binarize=False,
            pad_size={"height": frame.input_size, "width": frame.input_size},
        )
        return upscaled[0][:, 0].float()
