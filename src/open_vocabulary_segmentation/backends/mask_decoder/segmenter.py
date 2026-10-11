from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import torch
from open_vocabulary_detector import ImageSize
from PIL import Image

from ...array_types import BoolArray, FloatArray
from ...object_prompt import MaskLogits, ObjectPrompt
from ...options import MaskOutput
from ...promptable import EncodedImage, ModelFrame, PromptableSegmenter, PromptableSegmenterSettings
from ...result import ObjectSegmentationResult
from ...runtime import TorchRuntime
from ..checkpoint_config import CheckpointConfig
from .decoded_masks import DecodedMasks
from .preprocessed_images import PreprocessedImages
from .prompt_mask import PromptMask
from .prompt_signature import PromptSignature
from .prompt_tensors import PromptTensors


class MaskDecoderSegmenter(PromptableSegmenter):
    """
    Base of SAM-family segmenters: an image encoder run once per image and a light prompt encoder and mask decoder
    run per prompt.

    Prompts of one image are grouped by ``PromptSignature`` and every group is decoded in passes of at most
    ``settings.decode_batch_size`` prompts without padding, so batching never changes a mask; prompts with a mask
    are decoded one at a time. ``MaskOutput`` selects the single-mask or three-mask head per group. Mask logits are
    upscaled to the image ``UPSCALE_CHUNK_SIZE`` masks at a time, binarized at ``MASK_THRESHOLD`` and scored for
    stability, the IoU of the masks binarized ``STABILITY_OFFSET`` above and below the threshold.

    The mask prompt and the low-resolution logits are ``MASK_TO_EMBEDDING_RATIO`` times the side of the main image
    embedding (256 for SAM and SAM 2, 288 for SAM 3). Subclasses provide the model-specific steps through
    ``_preprocess``, ``_embed``, ``_decode`` and ``_upscale``; the main embedding must be the last feature tensor.
    """

    MASK_THRESHOLD: ClassVar[float] = 0.0
    STABILITY_OFFSET: ClassVar[float] = 1.0
    UPSCALE_CHUNK_SIZE: ClassVar[int] = 8
    MASK_TO_EMBEDDING_RATIO: ClassVar[int] = 4

    def __init__(self, settings: PromptableSegmenterSettings) -> None:
        """
        Check the checkpoint and prepare the runtime.

        Parameters
        ----------
        settings : PromptableSegmenterSettings
            Checkpoint, batching, device and precision.

        Raises
        ------
        ValueError
            If the checkpoint is not one of the backend's model types.
        """
        super().__init__(settings)
        CheckpointConfig.require_model_type(settings.weights_path, self.BACKEND.model_types)
        self._runtime: TorchRuntime = TorchRuntime(settings.device, settings.precision)

    @abstractmethod
    def _preprocess(self, images: Sequence[Image.Image]) -> PreprocessedImages:
        """
        Resize and normalize RGB images for the image encoder.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images.

        Returns
        -------
        PreprocessedImages
            Pixel values and frames.
        """

    @abstractmethod
    def _embed(self, pixel_values: torch.Tensor) -> list[tuple[torch.Tensor, ...]]:
        """
        Run the image encoder once over a mini-batch.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Preprocessed images on the runtime device and dtype, shape (B, C, S, S).

        Returns
        -------
        list[tuple[torch.Tensor, ...]]
            Features of each image with batch size 1, in input order.
        """

    @abstractmethod
    def _decode(self, features: tuple[torch.Tensor, ...], tensors: PromptTensors, is_multimask: bool) -> DecodedMasks:
        """
        Run the prompt encoder and mask decoder once.

        Parameters
        ----------
        features : tuple[torch.Tensor, ...]
            Features of one image.
        tensors : PromptTensors
            Prompts of one signature.
        is_multimask : bool
            Whether to use the three-mask head.

        Returns
        -------
        DecodedMasks
            Logits and predicted IoU as float32 on the CPU.
        """

    @abstractmethod
    def _upscale(self, low_resolution_logits: torch.Tensor, frame: ModelFrame) -> torch.Tensor:
        """
        Resize low-resolution logits to the source image without binarizing them.

        Parameters
        ----------
        low_resolution_logits : torch.Tensor
            Logits as float32 on the CPU, shape (M, h, w).
        frame : ModelFrame
            Placement of the image inside the model input.

        Returns
        -------
        torch.Tensor
            Logits over the source image as float32 on the CPU, shape (M, H, W).
        """

    def _encode_mini_batch(self, images: Sequence[Image.Image]) -> list[EncodedImage]:
        preprocessed: PreprocessedImages = self._preprocess(images)
        with torch.inference_mode():
            features: list[tuple[torch.Tensor, ...]] = self._embed(
                self._runtime.to_model_input(preprocessed.pixel_values)
            )
        return [
            EncodedImage(
                features=image_features,
                frame=frame,
                backend=self.BACKEND,
                weights_path=self.settings.weights_path,
            )
            for image_features, frame in zip(features, preprocessed.frames, strict=True)
        ]

    def _segment_encoded(
        self, encoding: EncodedImage, prompts: Sequence[ObjectPrompt], mask_output: MaskOutput
    ) -> ObjectSegmentationResult:
        if not prompts:
            return ObjectSegmentationResult.empty(prompts, encoding.image_size)
        groups: dict[PromptSignature, list[int]] = {}
        for prompt_index, prompt in enumerate(prompts):
            groups.setdefault(PromptSignature.of(prompt), []).append(prompt_index)
        prompt_masks: list[PromptMask] = []
        for signature, prompt_indices in groups.items():
            is_multimask: bool = mask_output.is_multimask(signature.is_ambiguous)
            pass_size: int = self.settings.decode_batch_size if signature.is_batchable else 1
            for start in range(0, len(prompt_indices), pass_size):
                pass_indices: list[int] = prompt_indices[start : start + pass_size]
                tensors: PromptTensors = PromptTensors.build(
                    [prompts[index] for index in pass_indices],
                    encoding.frame,
                    self._mask_input_size_of(encoding),
                    self._runtime,
                )
                with torch.inference_mode():
                    decoded: DecodedMasks = self._decode(encoding.features, tensors, is_multimask)
                if is_multimask and not mask_output.is_every_candidate_kept:
                    decoded = decoded.best()
                prompt_masks.extend(self._prompt_masks_of(decoded, pass_indices, encoding))
        prompt_masks.sort(key=lambda prompt_mask: (prompt_mask.prompt_index, prompt_mask.candidate_index))
        image_size: ImageSize = encoding.image_size
        return ObjectSegmentationResult(
            masks=np.stack([prompt_mask.mask for prompt_mask in prompt_masks]).reshape(
                -1, image_size.height, image_size.width
            ),
            scores=np.array([prompt_mask.score for prompt_mask in prompt_masks], dtype=np.float64),
            stability_scores=np.array([prompt_mask.stability_score for prompt_mask in prompt_masks], dtype=np.float64),
            prompt_indices=np.array([prompt_mask.prompt_index for prompt_mask in prompt_masks], dtype=np.int64),
            prompts=tuple(prompts),
            mask_logits=tuple(prompt_mask.mask_logits for prompt_mask in prompt_masks),
            image_size=image_size,
        )

    def _prompt_masks_of(
        self, decoded: DecodedMasks, prompt_indices: Sequence[int], encoding: EncodedImage
    ) -> list[PromptMask]:
        masks_per_prompt: int = decoded.masks_per_prompt
        flat_logits: torch.Tensor = decoded.low_resolution_logits.flatten(0, 1)
        flat_scores: FloatArray = decoded.iou_scores.flatten().clamp(0.0, 1.0).numpy().astype(np.float64)
        prompt_masks: list[PromptMask] = []
        for start in range(0, flat_logits.shape[0], self.UPSCALE_CHUNK_SIZE):
            chunk_logits: torch.Tensor = flat_logits[start : start + self.UPSCALE_CHUNK_SIZE]
            upscaled: torch.Tensor = self._upscale(chunk_logits, encoding.frame)
            masks: BoolArray = (upscaled > self.MASK_THRESHOLD).numpy()
            stability_scores: FloatArray = self._stability_scores(upscaled)
            for offset in range(chunk_logits.shape[0]):
                flat_index: int = start + offset
                prompt_masks.append(
                    PromptMask(
                        prompt_index=prompt_indices[flat_index // masks_per_prompt],
                        candidate_index=flat_index % masks_per_prompt,
                        mask=masks[offset],
                        score=float(flat_scores[flat_index]),
                        stability_score=float(stability_scores[offset]),
                        mask_logits=MaskLogits(
                            logits=chunk_logits[offset].numpy(),
                            backend=self.BACKEND,
                            weights_path=self.settings.weights_path,
                            image_size=encoding.image_size,
                        ),
                    )
                )
        return prompt_masks

    def _mask_input_size_of(self, encoding: EncodedImage) -> int:
        return self.MASK_TO_EMBEDDING_RATIO * int(encoding.features[-1].shape[-1])

    def _stability_scores(self, logits: torch.Tensor) -> FloatArray:
        intersections: torch.Tensor = (logits > self.MASK_THRESHOLD + self.STABILITY_OFFSET).sum(dim=(1, 2))
        unions: torch.Tensor = (logits > self.MASK_THRESHOLD - self.STABILITY_OFFSET).sum(dim=(1, 2))
        scores: torch.Tensor = torch.where(
            unions > 0, intersections.double() / unions.clamp(min=1).double(), torch.zeros_like(unions.double())
        )
        return scores.numpy().astype(np.float64)
