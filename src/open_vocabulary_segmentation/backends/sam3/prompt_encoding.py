from dataclasses import dataclass
from typing import cast

import torch
from open_vocabulary_detector import Prompt
from transformers.modeling_outputs import BaseModelOutputWithPooling


@dataclass(frozen=True, eq=False)
class PromptEncoding:
    """
    Text features of every query of one ``Prompt``, reused for every image batch segmented with it.

    SAM 3 encodes a text query independently of the image, so the text encoder runs once per prompt instead of once
    per query and image.

    Attributes
    ----------
    prompt : Prompt
        Prompt the encoding was built from.
    features : torch.Tensor
        Projected text features of every query, shape (Q, T, D).
    attention_mask : torch.Tensor
        Attention mask of every query, shape (Q, T).

    Raises
    ------
    ValueError
        If the rows do not match the queries of the prompt or the shapes disagree.
    """

    prompt: Prompt
    features: torch.Tensor
    attention_mask: torch.Tensor

    def __post_init__(self) -> None:
        query_count: int = len(self.prompt.queries)
        if self.features.ndim != 3 or self.features.shape[0] != query_count:
            raise ValueError(f"features must have shape ({query_count}, T, D). got {tuple(self.features.shape)}")
        if tuple(self.attention_mask.shape) != tuple(self.features.shape[:2]):
            raise ValueError(
                f"attention_mask must have shape {tuple(self.features.shape[:2])}. "
                + f"got {tuple(self.attention_mask.shape)}"
            )

    def is_for(self, prompt: Prompt) -> bool:
        """
        Check whether the encoding was built from a prompt.

        Parameters
        ----------
        prompt : Prompt
            Prompt to compare with.

        Returns
        -------
        bool
            True if ``prompt`` equals the prompt of the encoding.
        """
        return self.prompt == prompt

    def text_embeddings(self, query_id: int, batch_size: int) -> BaseModelOutputWithPooling:
        """
        Text features of one query for every image of a batch, in the form ``Sam3Model`` reads ``text_embeds``.

        Parameters
        ----------
        query_id : int
            Index of the query in ``prompt.queries``.
        batch_size : int
            Number of images.

        Returns
        -------
        BaseModelOutputWithPooling
            Output whose ``pooler_output`` holds the query features, shape (B, T, D).
        """
        return BaseModelOutputWithPooling(
            pooler_output=cast(torch.FloatTensor, self.features[query_id : query_id + 1].expand(batch_size, -1, -1))
        )

    def query_attention_mask(self, query_id: int, batch_size: int) -> torch.Tensor:
        """
        Attention mask of one query for every image of a batch.

        Parameters
        ----------
        query_id : int
            Index of the query in ``prompt.queries``.
        batch_size : int
            Number of images.

        Returns
        -------
        torch.Tensor
            Attention mask, shape (B, T).
        """
        return self.attention_mask[query_id : query_id + 1].expand(batch_size, -1)

    def to(self, device: torch.device) -> "PromptEncoding":
        """
        Move the tensors to a device.

        Parameters
        ----------
        device : torch.device
            Target device.

        Returns
        -------
        PromptEncoding
            Encoding with tensors on ``device``.
        """
        return PromptEncoding(
            prompt=self.prompt, features=self.features.to(device), attention_mask=self.attention_mask.to(device)
        )
