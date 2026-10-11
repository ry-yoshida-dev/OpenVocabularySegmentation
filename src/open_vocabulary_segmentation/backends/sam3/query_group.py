from dataclasses import dataclass

from ...exemplars import BoxExemplars
from .prompt_encoding import PromptEncoding


@dataclass(frozen=True, eq=False)
class QueryGroup:
    """
    Images of a mini-batch that SAM 3 decodes together for one query: same text, with or without exemplars.

    Attributes
    ----------
    image_indices : tuple[int, ...]
        Indices of the images in the mini-batch.
    text_encoding : PromptEncoding
        Encoding holding the text the decoders read.
    text_index : int
        Index of the text in ``text_encoding``.
    exemplars : tuple[BoxExemplars, ...] | None
        Exemplars of each image of the group; ``None`` for a group decoded from the text alone.

    Raises
    ------
    ValueError
        If the group is empty or the exemplars do not match the images.
    """

    image_indices: tuple[int, ...]
    text_encoding: PromptEncoding
    text_index: int
    exemplars: tuple[BoxExemplars, ...] | None

    def __post_init__(self) -> None:
        if not self.image_indices:
            raise ValueError("a query group needs at least one image.")
        if self.exemplars is not None and len(self.exemplars) != len(self.image_indices):
            raise ValueError(f"expected exemplars for {len(self.image_indices)} images. got {len(self.exemplars)}")
