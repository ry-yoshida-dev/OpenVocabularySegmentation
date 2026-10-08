import pytest
import torch

from open_vocabulary_segmentation import Prompt
from open_vocabulary_segmentation.backends.sam3 import PromptEncoding

PROMPT: Prompt = Prompt.from_texts({"car": ("car", "suv"), "dog": ("dog",)})


def build_encoding() -> PromptEncoding:
    features: torch.Tensor = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    attention_mask: torch.Tensor = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]])
    return PromptEncoding(prompt=PROMPT, features=features, attention_mask=attention_mask)


def test_query_features_are_repeated_for_every_image() -> None:
    encoding: PromptEncoding = build_encoding()
    pooler_output = encoding.text_embeddings(query_id=1, batch_size=3).pooler_output
    assert pooler_output is not None
    assert tuple(pooler_output.shape) == (3, 4, 2)
    assert torch.equal(pooler_output[2], encoding.features[1])
    assert encoding.query_attention_mask(query_id=2, batch_size=2).tolist() == [[1, 0, 0, 0], [1, 0, 0, 0]]


def test_encoding_is_reused_for_an_equal_prompt_only() -> None:
    encoding: PromptEncoding = build_encoding()
    assert encoding.is_for(Prompt.from_texts({"car": ("car", "suv"), "dog": ("dog",)}))
    assert not encoding.is_for(Prompt.from_texts({"car": ("car",), "dog": ("dog",)}))


def test_rows_must_match_the_queries() -> None:
    with pytest.raises(ValueError, match="features must have shape"):
        PromptEncoding(prompt=PROMPT, features=torch.zeros(2, 4, 2), attention_mask=torch.ones(2, 4))
    with pytest.raises(ValueError, match="attention_mask must have shape"):
        PromptEncoding(prompt=PROMPT, features=torch.zeros(3, 4, 2), attention_mask=torch.ones(3, 5))
