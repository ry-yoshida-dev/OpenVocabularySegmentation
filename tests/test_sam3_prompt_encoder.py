from typing import cast

import numpy as np
from geometry import Box2DFormat, Boxes2D
from PIL import Image
from stub_sam3 import StubSam3
from transformers import Sam3Model, Sam3Processor

from open_vocabulary_segmentation import Device, Precision, Prompt, TextQuery, VisualQuery, VisualReference
from open_vocabulary_segmentation.backends.sam3 import PromptEncoder, PromptEncoding
from open_vocabulary_segmentation.runtime import TorchRuntime


def reference_with(box_count: int) -> VisualReference:
    xyxy: list[list[float]] = [[0.0, 0.0, 4.0 + index, 4.0 + index] for index in range(box_count)]
    return VisualReference(
        image=Image.new("RGB", (16, 16)),
        boxes=Boxes2D.register(value=np.array(xyxy, dtype=np.float64), box2d_format=Box2DFormat.XYXY),
    )


def build_encoder(stub: StubSam3) -> PromptEncoder:
    return PromptEncoder(cast(Sam3Model, stub), cast(Sam3Processor, stub), TorchRuntime(Device.CPU, Precision.FLOAT32))


def test_visual_query_reads_the_visual_text_and_the_geometry_of_its_references() -> None:
    stub: StubSam3 = StubSam3()
    references: tuple[VisualReference, ...] = (reference_with(2), reference_with(1))
    prompt: Prompt = Prompt({"car": (TextQuery("car"), VisualQuery(references)), "dog": (TextQuery("dog"),)})
    encoding: PromptEncoding = build_encoder(stub).encode(prompt)
    text_count: int = StubSam3.TEXT_TOKEN_COUNT
    geometry_start: int = PromptEncoder.TEXT_TOKEN_COUNT
    geometry_count: int = (2 + 1) + (1 + 1)
    assert tuple(encoding.features.shape) == (3, geometry_start + geometry_count, StubSam3.HIDDEN_SIZE)
    assert encoding.attention_mask.sum(dim=1).tolist() == [text_count, text_count + geometry_count, text_count]
    assert encoding.attention_mask[1, geometry_start:].tolist() == [1] * geometry_count
    assert encoding.features[0, :text_count].unique().tolist() == [0.0]
    assert encoding.features[1, :text_count].unique().tolist() == [1.0]
    assert encoding.features[2, :text_count].unique().tolist() == [2.0]
    assert encoding.features[1, geometry_start : geometry_start + 3].unique().tolist() == [-1.0]
    assert encoding.features[1, geometry_start + 3 :].unique().tolist() == [-2.0]
    assert encoding.features[0, geometry_start:].abs().sum().item() == 0.0


def test_geometry_is_reused_while_consecutive_prompts_use_the_reference() -> None:
    stub: StubSam3 = StubSam3()
    encoder: PromptEncoder = build_encoder(stub)
    reference: VisualReference = reference_with(1)
    encoder.encode(Prompt({"car": (TextQuery("car"), VisualQuery((reference,)))}))
    encoder.encode(Prompt({"car": (TextQuery("suv"), VisualQuery((reference,)))}))
    assert stub.geometry_call_count == 1
    encoder.encode(Prompt.from_class_names(("car",)))
    encoder.encode(Prompt({"car": (VisualQuery((reference,)),)}))
    assert stub.geometry_call_count == 2


def test_text_prompt_keeps_the_text_token_layout() -> None:
    encoding: PromptEncoding = build_encoder(StubSam3()).encode(Prompt.from_texts({"car": ("car", "suv")}))
    assert tuple(encoding.attention_mask.shape) == (2, PromptEncoder.TEXT_TOKEN_COUNT)
