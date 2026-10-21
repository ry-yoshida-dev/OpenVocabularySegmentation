from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class ReferenceGeometry:
    """
    Geometry prompt tokens of the boxes of one visual reference, encoded on its reference image.

    Attributes
    ----------
    features : torch.Tensor
        One token per box plus the CLS token of the geometry encoder, shape (G, D).
    attention_mask : torch.Tensor
        ``1`` for valid tokens, shape (G,).

    Raises
    ------
    ValueError
        If the shapes disagree.
    """

    features: torch.Tensor
    attention_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"features must have shape (G, D). got {tuple(self.features.shape)}")
        if tuple(self.attention_mask.shape) != (self.features.shape[0],):
            raise ValueError(
                f"attention_mask must have shape ({self.features.shape[0]},). got {tuple(self.attention_mask.shape)}"
            )

    @property
    def token_count(self) -> int:
        """
        Number of tokens.

        Returns
        -------
        int
            Boxes of the reference plus one.
        """
        return int(self.features.shape[0])
