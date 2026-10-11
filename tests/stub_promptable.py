from collections.abc import Sequence
from typing import ClassVar

import pytest
import torch
from open_vocabulary_detector import Device
from PIL import Image
from torch.nn import functional

from open_vocabulary_segmentation import PromptableBackend, PromptableSegmenterSettings
from open_vocabulary_segmentation.backends.checkpoint_config import CheckpointConfig
from open_vocabulary_segmentation.backends.mask_decoder import (
    DecodedMasks,
    MaskDecoderSegmenter,
    PointLabel,
    PreprocessedImages,
    PromptTensors,
)
from open_vocabulary_segmentation.promptable import ModelFrame

INPUT_SIZE: int = 16
EMBEDDING_SIZE: int = 4
INSIDE_LOGIT: float = 10.0
CANDIDATE_SCORES: tuple[float, float, float] = (0.5, 0.9, 0.7)
SINGLE_SCORE: float = 0.8


def patch_checkpoint(monkeypatch: pytest.MonkeyPatch, model_type: str) -> None:
    def read_model_type(weights_path: str) -> str:
        return model_type

    monkeypatch.setattr(CheckpointConfig, "read_model_type", staticmethod(read_model_type))


def build_promptable_settings(
    backend: PromptableBackend = PromptableBackend.SAM2, decode_batch_size: int = 64, weights_path: str = "stub"
) -> PromptableSegmenterSettings:
    return PromptableSegmenterSettings(
        backend=backend,
        weights_path=weights_path,
        batch_size=2,
        decode_batch_size=decode_batch_size,
        device=Device.CPU,
    )


class StubPromptableSegmenter(MaskDecoderSegmenter):
    """
    Mask decoder that paints boxes and 3x3 squares around positive points into its logits, so masks are predictable.

    The image is stretched over a 16x16 input; the low-resolution logits are the 16x16 input frame itself. With the
    three-mask head the candidates get the scores of ``CANDIDATE_SCORES``.
    """

    BACKEND: ClassVar[PromptableBackend] = PromptableBackend.SAM2

    def __init__(self, settings: PromptableSegmenterSettings) -> None:
        super().__init__(settings)
        self.embedded_batch_sizes: list[int] = []
        self.decoded_prompt_counts: list[int] = []
        self.multimask_flags: list[bool] = []

    def _preprocess(self, images: Sequence[Image.Image]) -> PreprocessedImages:
        original_sizes: torch.Tensor = torch.tensor([[image.height, image.width] for image in images])
        return PreprocessedImages.stretched(torch.zeros(len(images), 3, INPUT_SIZE, INPUT_SIZE), original_sizes)

    def _embed(self, pixel_values: torch.Tensor) -> list[tuple[torch.Tensor, ...]]:
        self.embedded_batch_sizes.append(int(pixel_values.shape[0]))
        return [(torch.zeros(1, 2, EMBEDDING_SIZE, EMBEDDING_SIZE),) for _ in range(pixel_values.shape[0])]

    def _decode(self, features: tuple[torch.Tensor, ...], tensors: PromptTensors, is_multimask: bool) -> DecodedMasks:
        prompt_count: int = self._prompt_count(tensors)
        self.decoded_prompt_counts.append(prompt_count)
        self.multimask_flags.append(is_multimask)
        logits: torch.Tensor = torch.full((prompt_count, INPUT_SIZE, INPUT_SIZE), -INSIDE_LOGIT)
        if tensors.masks is not None:
            logits = functional.interpolate(tensors.masks.float(), size=(INPUT_SIZE, INPUT_SIZE))[:, 0].clone()
        if tensors.boxes is not None:
            for index, (x_min, y_min, x_max, y_max) in enumerate(tensors.boxes[0].round().long().tolist()):
                logits[index, y_min:y_max, x_min:x_max] = INSIDE_LOGIT
        if tensors.points is not None and tensors.labels is not None:
            for index in range(prompt_count):
                for (x, y), label in zip(
                    tensors.points[0, index].round().long().tolist(), tensors.labels[0, index].tolist(), strict=True
                ):
                    if label == PointLabel.POSITIVE:
                        logits[index, max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = INSIDE_LOGIT
        if is_multimask:
            return DecodedMasks(
                low_resolution_logits=logits[:, None].repeat(1, 3, 1, 1),
                iou_scores=torch.tensor([CANDIDATE_SCORES] * prompt_count),
            )
        return DecodedMasks(
            low_resolution_logits=logits[:, None], iou_scores=torch.full((prompt_count, 1), SINGLE_SCORE)
        )

    def _upscale(self, low_resolution_logits: torch.Tensor, frame: ModelFrame) -> torch.Tensor:
        return functional.interpolate(
            low_resolution_logits[:, None], size=(frame.image_size.height, frame.image_size.width), mode="nearest"
        )[:, 0]

    @staticmethod
    def _prompt_count(tensors: PromptTensors) -> int:
        for tensor in (tensors.points, tensors.boxes):
            if tensor is not None:
                return int(tensor.shape[1])
        if tensors.masks is not None:
            return int(tensors.masks.shape[0])
        raise ValueError("prompt tensors are empty.")


class StubSamSegmenter(StubPromptableSegmenter):
    BACKEND: ClassVar[PromptableBackend] = PromptableBackend.SAM
