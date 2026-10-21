from collections.abc import Sequence
from dataclasses import replace
from typing import ClassVar, cast

import numpy as np
import torch
from open_vocabulary_detector import ImageSize, Prompt, PromptKind
from PIL import Image
from transformers import BatchFeature, Sam3LiteTextModel, Sam3Model, Sam3Processor

from ...array_types import BoolArray, Float32Array, FloatArray, IntArray
from ...exemplars import BoxExemplars, ImageExemplars
from ...image import ImageInput, RgbImage
from ...options import SegmenterBackend
from ...result import SegmentationResult, SemanticSegmentationResult
from ...runtime import TorchRuntime
from ...segmenter import OpenVocabularySegmenter
from ...settings import SegmenterSettings
from ..checkpoint_config import CheckpointConfig
from .exemplar_tensors import ExemplarTensors
from .model_outputs import Sam3SegmentationOutput, Sam3VisionOutput
from .prediction import Sam3Prediction
from .prompt_encoder import PromptEncoder
from .prompt_encoding import PromptEncoding
from .query_group import QueryGroup


class Sam3Segmenter(OpenVocabularySegmenter):
    """
    SAM 3 promptable concept segmentation backed by ``transformers.Sam3Model`` (or ``Sam3LiteTextModel``).

    SAM 3 finds and masks every instance of a concept on its own. A concept is a text query or a visual query, whose
    reference boxes are encoded on their reference images (``PromptEncoder``), optionally with box exemplars drawn on
    the image (``ImageExemplars``): positive boxes show the class, negative boxes show look-alikes to exclude, and a
    class may also be defined by its boxes alone (``BoxExemplars.is_text_ignored``), which SAM 3 reads as the text
    ``PromptEncoder.VISUAL_TEXT``. Besides instances, the semantic head gives a probability map per class
    (``segment_semantic_images``).

    The images of a mini-batch are encoded once, the queries of a prompt are encoded once and reused while the same
    prompt is segmented, and the detection and mask decoders run once per query over the images without
    exemplars of its class and once over the images with exemplars, whose box counts are padded with ignored
    boxes. An instance scores the product of its query match and the presence of the concept in the image. Queries
    run independently, so one object may be found by several queries; the class-wise NMS merges those of one class.

    When no instance of a query passes the threshold, the ``transformers``
    post-processing returns its masks at model resolution, so empty results are rebuilt at the image size.
    """

    BACKENDS: ClassVar[frozenset[SegmenterBackend]] = frozenset({SegmenterBackend.SAM3})
    MODEL_TYPES: ClassVar[frozenset[str]] = frozenset({"sam3", "sam3_video", "sam3_lite_text"})
    LITE_TEXT_MODEL_TYPES: ClassVar[frozenset[str]] = frozenset({"sam3_lite_text"})
    MASK_THRESHOLD: ClassVar[float] = 0.5

    def __init__(self, settings: SegmenterSettings) -> None:
        """
        Check the checkpoint, then load the processor and model.

        Parameters
        ----------
        settings : SegmenterSettings
            SAM 3 checkpoint, batching, device, precision and thresholds.

        Raises
        ------
        ValueError
            If the checkpoint is not one of ``MODEL_TYPES``.
        """
        super().__init__(settings)
        model_type: str = CheckpointConfig.require_model_type(settings.weights_path, self.MODEL_TYPES)
        self._runtime: TorchRuntime = TorchRuntime(settings.device, settings.precision)
        self._processor: Sam3Processor = Sam3Processor.from_pretrained(settings.weights_path)
        self._model: Sam3Model | Sam3LiteTextModel = (
            Sam3LiteTextModel.from_pretrained(settings.weights_path)
            if model_type in self.LITE_TEXT_MODEL_TYPES
            else Sam3Model.from_pretrained(settings.weights_path)
        )
        self._runtime.prepare_model(self._model)
        self._prompt_encoder: PromptEncoder = PromptEncoder(self._model, self._processor, self._runtime)
        self._active_encoding: PromptEncoding | None = None
        self._visual_only_encoding: PromptEncoding | None = None

    @property
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        """
        Query kinds the segmenter accepts.

        Returns
        -------
        frozenset[PromptKind]
            ``TEXT`` and ``VISUAL``; boxes on the segmented images refine concepts through ``ImageExemplars``.
        """
        return frozenset({PromptKind.TEXT, PromptKind.VISUAL})

    @property
    def is_exemplar_supported(self) -> bool:
        """
        Whether the segmenter reads box exemplars next to the prompt.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def segment_semantic(
        self, image: ImageInput, prompt: Prompt, exemplars: ImageExemplars | None = None
    ) -> SemanticSegmentationResult:
        """
        Probability map of every class of the prompt in a single image.

        Parameters
        ----------
        image : ImageInput
            Input image, a PIL image or an ``image_container`` container of any channel order.
        prompt : Prompt
            Classes to segment.
        exemplars : ImageExemplars | None, optional
            Box exemplars drawn on ``image``.

        Returns
        -------
        SemanticSegmentationResult
            Per-class probabilities of the image.
        """
        return self.segment_semantic_images([image], prompt, None if exemplars is None else [exemplars])[0]

    def segment_semantic_images(
        self,
        images: Sequence[ImageInput],
        prompt: Prompt,
        exemplars: Sequence[ImageExemplars | None] | None = None,
    ) -> list[SemanticSegmentationResult]:
        """
        Probability map of every class of the prompt in many images, processed in mini-batches.

        Parameters
        ----------
        images : Sequence[ImageInput]
            Input images of arbitrary size.
        prompt : Prompt
            Classes to segment.
        exemplars : Sequence[ImageExemplars | None] | None, optional
            Box exemplars of each image.

        Raises
        ------
        ValueError
            If ``images`` is empty, the prompt uses an unsupported query kind, or the exemplars do not match the
            images.
        KeyError
            If exemplars refer to a class that is not in the prompt.

        Returns
        -------
        list[SemanticSegmentationResult]
            One result per image, in input order.
        """
        if not images:
            raise ValueError("images must contain at least one image.")
        self._validate_prompt(prompt)
        image_exemplars: list[ImageExemplars | None] = self._exemplars_per_image(images, exemplars)
        results: list[SemanticSegmentationResult] = []
        for start in range(0, len(images), self.batch_size):
            rgb_images: list[Image.Image] = [
                RgbImage.from_input(image) for image in images[start : start + self.batch_size]
            ]
            mini_batch_exemplars: list[ImageExemplars | None] = image_exemplars[start : start + self.batch_size]
            for rgb_image, exemplars_of_image in zip(rgb_images, mini_batch_exemplars, strict=True):
                if exemplars_of_image is not None:
                    exemplars_of_image.validate_for(prompt, ImageSize.from_image(rgb_image))
            results.extend(
                self._predict(rgb_images, prompt, mini_batch_exemplars, is_semantic_requested=True).semantics
            )
        return results

    def _segment_mini_batch(
        self, images: Sequence[Image.Image], prompt: Prompt, exemplars: Sequence[ImageExemplars | None]
    ) -> list[SegmentationResult]:
        return list(self._predict(images, prompt, exemplars, is_semantic_requested=False).instances)

    def _predict(
        self,
        images: Sequence[Image.Image],
        prompt: Prompt,
        exemplars: Sequence[ImageExemplars | None],
        is_semantic_requested: bool,
    ) -> Sam3Prediction:
        encoding: PromptEncoding = self._encoding_of(prompt)
        image_inputs: BatchFeature = self._processor.image_processor(images=list(images), return_tensors="pt")
        image_sizes: list[ImageSize] = [ImageSize.from_image(image) for image in images]
        prediction: Sam3Prediction = Sam3Prediction.start(prompt, image_sizes, is_semantic_requested)
        with torch.inference_mode():
            vision: Sam3VisionOutput = self._model.get_vision_features(
                pixel_values=cast(torch.FloatTensor, self._runtime.to_model_input(image_inputs["pixel_values"]))
            )
            for query_id, class_id in enumerate(prompt.query_class_ids):
                for group in self._query_groups(prompt, query_id, encoding, exemplars):
                    self._run_group(vision, group, query_id, class_id, prediction)
        return prediction

    def _query_groups(
        self,
        prompt: Prompt,
        query_id: int,
        encoding: PromptEncoding,
        exemplars: Sequence[ImageExemplars | None],
    ) -> list[QueryGroup]:
        class_id: int = prompt.query_class_ids[query_id]
        class_name: str = prompt.class_names[class_id]
        is_first_query_of_class: bool = prompt.query_class_ids.index(class_id) == query_id
        class_exemplars: list[BoxExemplars | None] = [
            None if image_exemplars is None else image_exemplars.of_class(class_name) for image_exemplars in exemplars
        ]
        plain_indices: tuple[int, ...] = tuple(index for index, entry in enumerate(class_exemplars) if entry is None)
        text_entries: list[tuple[int, BoxExemplars]] = [
            (index, entry)
            for index, entry in enumerate(class_exemplars)
            if entry is not None and not entry.is_text_ignored
        ]
        visual_entries: list[tuple[int, BoxExemplars]] = [
            (index, entry)
            for index, entry in enumerate(class_exemplars)
            if entry is not None and entry.is_text_ignored and is_first_query_of_class
        ]
        groups: list[QueryGroup] = []
        if plain_indices:
            groups.append(
                QueryGroup(image_indices=plain_indices, text_encoding=encoding, text_index=query_id, exemplars=None)
            )
        if text_entries:
            groups.append(
                QueryGroup(
                    image_indices=tuple(index for index, _ in text_entries),
                    text_encoding=encoding,
                    text_index=query_id,
                    exemplars=tuple(entry for _, entry in text_entries),
                )
            )
        if visual_entries:
            groups.append(
                QueryGroup(
                    image_indices=tuple(index for index, _ in visual_entries),
                    text_encoding=self._visual_only_encoding_of(),
                    text_index=0,
                    exemplars=tuple(entry for _, entry in visual_entries),
                )
            )
        return groups

    def _run_group(
        self,
        vision: Sam3VisionOutput,
        group: QueryGroup,
        query_id: int,
        class_id: int,
        prediction: Sam3Prediction,
    ) -> None:
        exemplar_tensors: ExemplarTensors | None = (
            None
            if group.exemplars is None
            else ExemplarTensors.build(
                group.exemplars, [prediction.image_sizes[index] for index in group.image_indices], self._runtime
            )
        )
        outputs: Sam3SegmentationOutput = self._run_decoders(
            self._select_images(vision, group.image_indices, len(prediction.image_sizes)),
            group.text_encoding,
            group.text_index,
            exemplar_tensors,
        )
        self._collect(outputs, group.image_indices, query_id, class_id, prediction)

    def _run_decoders(
        self,
        vision: Sam3VisionOutput,
        text_encoding: PromptEncoding,
        text_query_id: int,
        exemplar_tensors: ExemplarTensors | None,
    ) -> Sam3SegmentationOutput:
        batch_size: int = int(vision.fpn_hidden_states[0].shape[0])
        outputs: Sam3SegmentationOutput = self._model(
            vision_embeds=vision,
            text_embeds=text_encoding.text_embeddings(text_query_id, batch_size),
            attention_mask=text_encoding.query_attention_mask(text_query_id, batch_size),
            input_boxes=None if exemplar_tensors is None else exemplar_tensors.boxes,
            input_boxes_labels=None if exemplar_tensors is None else exemplar_tensors.labels,
        )
        return outputs

    def _collect(
        self,
        outputs: Sam3SegmentationOutput,
        indices: Sequence[int],
        query_id: int,
        class_id: int,
        prediction: Sam3Prediction,
    ) -> None:
        image_sizes: tuple[ImageSize, ...] = prediction.image_sizes
        target_sizes: list[tuple[int, int]] = [
            (image_sizes[index].height, image_sizes[index].width) for index in indices
        ]
        instances: list[dict[str, torch.Tensor]] = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=self.thresholds.confidence_threshold,
            mask_threshold=self.MASK_THRESHOLD,
            target_sizes=target_sizes,
        )
        for image_index, image_instances, (height, width) in zip(indices, instances, target_sizes, strict=True):
            confidences: FloatArray = image_instances["scores"].float().cpu().numpy().astype(np.float64)
            masks: BoolArray = (
                image_instances["masks"].bool().cpu().numpy().reshape(-1, height, width)
                if len(confidences)
                else np.zeros((0, height, width), dtype=np.bool_)
            )
            query_ids: IntArray = np.full(len(confidences), query_id, dtype=np.int64)
            prediction.add_instances(image_index, masks, confidences, query_ids)
        if prediction.is_semantic_requested:
            semantics: list[object] = self._processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes, return_segmentation_scores=True
            )
            for image_index, semantic in zip(indices, semantics, strict=True):
                scores: object = getattr(semantic, "segmentation_scores", None)
                if not isinstance(scores, torch.Tensor):
                    raise TypeError(f"unexpected semantic segmentation output: {type(semantic)}")
                probabilities: Float32Array = (
                    scores.float()
                    .cpu()
                    .numpy()
                    .reshape(image_sizes[image_index].height, image_sizes[image_index].width)
                )
                prediction.add_semantic(image_index, class_id, probabilities)

    @staticmethod
    def _select_images(vision: Sam3VisionOutput, indices: Sequence[int], image_count: int) -> Sam3VisionOutput:
        if list(indices) == list(range(image_count)):
            return vision
        index_tensor: torch.Tensor = torch.tensor(
            list(indices), dtype=torch.long, device=vision.fpn_hidden_states[0].device
        )
        last_hidden_state: torch.Tensor | None = vision.last_hidden_state
        return replace(
            vision,
            last_hidden_state=(
                None
                if last_hidden_state is None
                else cast(torch.FloatTensor, last_hidden_state.index_select(0, index_tensor))
            ),
            fpn_hidden_states=tuple(
                cast(torch.FloatTensor, features.index_select(0, index_tensor)) for features in vision.fpn_hidden_states
            ),
            fpn_position_encoding=tuple(
                cast(torch.FloatTensor, encoding.index_select(0, index_tensor))
                for encoding in vision.fpn_position_encoding
            ),
        )

    def _encoding_of(self, prompt: Prompt) -> PromptEncoding:
        if self._active_encoding is None or not self._active_encoding.is_for(prompt):
            self._active_encoding = self._prompt_encoder.encode(prompt)
        return self._active_encoding

    def _visual_only_encoding_of(self) -> PromptEncoding:
        if self._visual_only_encoding is None:
            self._visual_only_encoding = PromptEncoder(self._model, self._processor, self._runtime).encode(
                Prompt.from_class_names((PromptEncoder.VISUAL_TEXT,))
            )
        return self._visual_only_encoding
