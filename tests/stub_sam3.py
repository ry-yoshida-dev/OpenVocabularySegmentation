from collections.abc import Sequence
from types import SimpleNamespace
from typing import ClassVar

import torch
from PIL import Image


class StubSam3:
    """
    SAM 3 detector and processor in one: text features encode the token count, geometry features the box count.

    Every text reads ``TEXT_TOKEN_COUNT`` valid tokens; a geometry encoding has one token per box plus the CLS token,
    filled with the number of the call so that cached geometry is recognized.
    """

    HIDDEN_SIZE: ClassVar[int] = 4
    TEXT_TOKEN_COUNT: ClassVar[int] = 3

    def __init__(self) -> None:
        self.geometry_call_count: int = 0
        self.tokenizer: StubSam3 = self
        self.image_processor: StubSam3 = self

    def __call__(
        self,
        texts: Sequence[str] | None = None,
        images: Sequence[Image.Image] | None = None,
        return_tensors: str = "pt",
        padding: str = "max_length",
        max_length: int = 32,
    ) -> dict[str, torch.Tensor]:
        if images is not None:
            return {"pixel_values": torch.zeros(len(images), 3, 8, 8)}
        if texts is None:
            raise ValueError("texts or images are needed.")
        attention_mask: torch.Tensor = torch.zeros(len(texts), max_length, dtype=torch.long)
        attention_mask[:, : self.TEXT_TOKEN_COUNT] = 1
        input_ids: torch.Tensor = torch.arange(len(texts)).unsqueeze(1).expand(-1, max_length)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        features: torch.Tensor = input_ids.unsqueeze(-1).expand(-1, -1, self.HIDDEN_SIZE).float()
        return SimpleNamespace(pooler_output=features)

    def get_vision_features(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        features: tuple[torch.Tensor, ...] = (torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 1, 1))
        return SimpleNamespace(fpn_hidden_states=features, fpn_position_encoding=features)

    def geometry_encoder(
        self,
        box_embeddings: torch.Tensor,
        box_mask: torch.Tensor,
        box_labels: torch.Tensor,
        img_feats: tuple[torch.Tensor, ...],
        img_pos_embeds: tuple[torch.Tensor, ...],
    ) -> SimpleNamespace:
        self.geometry_call_count += 1
        token_count: int = int(box_embeddings.shape[1]) + 1
        features: torch.Tensor = torch.full((1, token_count, self.HIDDEN_SIZE), float(-self.geometry_call_count))
        return SimpleNamespace(last_hidden_state=features, attention_mask=torch.ones(1, token_count, dtype=torch.bool))
