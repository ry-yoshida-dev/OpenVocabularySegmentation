from collections.abc import Sequence
from typing import ClassVar, cast

import torch
from open_vocabulary_detector import ImageSize, Prompt, TextQuery, VisualQuery, VisualReference
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Sam3LiteTextModel, Sam3Model, Sam3Processor, Sam3VideoProcessor

from ...exemplars import BoxExemplars
from ...image import RgbImage
from ...runtime import TorchRuntime
from .exemplar_tensors import ExemplarTensors
from .model_outputs import Sam3VisionOutput
from .prompt_encoding import PromptEncoding
from .reference_geometry import ReferenceGeometry


class PromptEncoder:
    """
    Encodes the text and visual queries of a prompt into the prompt features the SAM 3 detector reads.

    A text query reads its phrase. A visual query reads the text ``VISUAL_TEXT`` followed by the geometry tokens of
    the boxes of each of its references, encoded by the SAM 3 geometry encoder on the reference image itself. SAM 3
    reads box exemplars of the segmented image the same way, so the concept shown on the reference images is found
    on any other image. Rows are padded to the longest query and the padding is masked out.

    The geometry of a reference is kept while consecutive prompts still use it, so editing the phrases of a prompt
    does not encode its reference images again. ``get_text_features`` returns the output of the text encoder with the
    projected features attached as ``pooler_output`` (``PROJECTED_TEXT_FEATURES_ATTRIBUTE``), which is not part of
    that output type, so the attribute is read dynamically.
    """

    VISUAL_TEXT: ClassVar[str] = "visual"
    TEXT_TOKEN_COUNT: ClassVar[int] = 32
    PROJECTED_TEXT_FEATURES_ATTRIBUTE: ClassVar[str] = "pooler_output"

    def __init__(
        self,
        model: Sam3Model | Sam3LiteTextModel,
        processor: Sam3Processor | Sam3VideoProcessor,
        runtime: TorchRuntime,
    ) -> None:
        """
        Parameters
        ----------
        model : Sam3Model | Sam3LiteTextModel
            Loaded SAM 3 detector.
        processor : Sam3Processor | Sam3VideoProcessor
            Processor providing the tokenizer and image processor of the checkpoint.
        runtime : TorchRuntime
            Device and dtype of the model.
        """
        self._model: Sam3Model | Sam3LiteTextModel = model
        self._processor: Sam3Processor | Sam3VideoProcessor = processor
        self._runtime: TorchRuntime = runtime
        self._geometries: dict[VisualReference, ReferenceGeometry] = {}

    def encode(self, prompt: Prompt) -> PromptEncoding:
        """
        Prompt features of every query of a prompt.

        Parameters
        ----------
        prompt : Prompt
            Classes with their text and visual queries.

        Returns
        -------
        PromptEncoding
            One padded row per query, in query-id order.
        """
        texts: tuple[str, ...] = tuple(self._text_of(query) for query in prompt.queries)
        distinct_texts: tuple[str, ...] = tuple(dict.fromkeys(texts))
        geometries: dict[VisualReference, ReferenceGeometry] = {}
        rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        with torch.inference_mode():
            text_features, text_mask = self._encode_texts(distinct_texts)
            for query, text in zip(prompt.queries, texts, strict=True):
                text_index: int = distinct_texts.index(text)
                row_features: list[torch.Tensor] = [text_features[text_index]]
                row_mask: list[torch.Tensor] = [text_mask[text_index]]
                if isinstance(query, VisualQuery):
                    for reference in query.references:
                        geometry: ReferenceGeometry = geometries.get(reference) or self._geometry_of(reference)
                        geometries[reference] = geometry
                        row_features.append(geometry.features)
                        row_mask.append(geometry.attention_mask)
                rows.append((torch.cat(row_features), torch.cat(row_mask)))
            encoding: PromptEncoding = self._padded(prompt, rows)
        self._geometries = geometries
        return encoding

    def _text_of(self, query: TextQuery | VisualQuery) -> str:
        match query:
            case TextQuery(text=text):
                return text
            case VisualQuery():
                return self.VISUAL_TEXT

    def _encode_texts(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        text_inputs: BatchEncoding = self._processor.tokenizer(
            list(texts), return_tensors="pt", padding="max_length", max_length=self.TEXT_TOKEN_COUNT
        )
        input_ids: torch.Tensor = self._runtime.to_device(text_inputs["input_ids"])
        attention_mask: torch.Tensor = self._runtime.to_device(text_inputs["attention_mask"])
        text_outputs: object = cast(
            object,
            self._model.get_text_features(input_ids=cast(torch.LongTensor, input_ids), attention_mask=attention_mask),
        )
        features: object = getattr(text_outputs, self.PROJECTED_TEXT_FEATURES_ATTRIBUTE, None)
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"unexpected text encoder output: {type(text_outputs)}")
        return features, attention_mask

    def _geometry_of(self, reference: VisualReference) -> ReferenceGeometry:
        cached: ReferenceGeometry | None = self._geometries.get(reference)
        if cached is not None:
            return cached
        image: Image.Image = RgbImage.from_input(reference.image)
        image_inputs: BatchFeature = self._processor.image_processor(images=[image], return_tensors="pt")
        exemplar_tensors: ExemplarTensors = ExemplarTensors.build(
            (BoxExemplars(positive_boxes=reference.boxes),), (ImageSize.from_image(image),), self._runtime
        )
        vision: Sam3VisionOutput = self._model.get_vision_features(
            pixel_values=cast(torch.FloatTensor, self._runtime.to_model_input(image_inputs["pixel_values"]))
        )
        geometry_outputs: object = cast(
            object,
            self._model.geometry_encoder(
                box_embeddings=exemplar_tensors.boxes,
                box_mask=exemplar_tensors.labels != ExemplarTensors.PADDING_LABEL,
                box_labels=exemplar_tensors.labels,
                img_feats=vision.fpn_hidden_states[:-1],
                img_pos_embeds=vision.fpn_position_encoding[:-1],
            ),
        )
        features: object = getattr(geometry_outputs, "last_hidden_state", None)
        mask: object = getattr(geometry_outputs, "attention_mask", None)
        if not isinstance(features, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError(f"unexpected geometry encoder output: {type(geometry_outputs)}")
        return ReferenceGeometry(features=features[0], attention_mask=mask[0].long())

    @staticmethod
    def _padded(prompt: Prompt, rows: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> PromptEncoding:
        token_count: int = max(int(features.shape[0]) for features, _ in rows)
        first_features: torch.Tensor = rows[0][0]
        features: torch.Tensor = first_features.new_zeros((len(rows), token_count, int(first_features.shape[1])))
        attention_mask: torch.Tensor = torch.zeros(
            (len(rows), token_count), dtype=torch.long, device=first_features.device
        )
        for index, (row_features, row_mask) in enumerate(rows):
            features[index, : row_features.shape[0]] = row_features
            attention_mask[index, : row_mask.shape[0]] = row_mask.long()
        return PromptEncoding(prompt=prompt, features=features, attention_mask=attention_mask)
