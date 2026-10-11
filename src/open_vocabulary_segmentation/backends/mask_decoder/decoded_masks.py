from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class DecodedMasks:
    """
    Raw output of one mask decoder pass.

    Attributes
    ----------
    low_resolution_logits : torch.Tensor
        Mask logits in the model input frame as float32 on the CPU, shape (N, K, h, w) with K masks per prompt.
    iou_scores : torch.Tensor
        Predicted IoU of every mask as float32 on the CPU, shape (N, K).

    Raises
    ------
    ValueError
        If the shapes disagree.
    """

    low_resolution_logits: torch.Tensor
    iou_scores: torch.Tensor

    def __post_init__(self) -> None:
        if self.low_resolution_logits.ndim != 4:
            raise ValueError(
                f"low_resolution_logits must have shape (N, K, h, w). got {tuple(self.low_resolution_logits.shape)}"
            )
        if tuple(self.iou_scores.shape) != tuple(self.low_resolution_logits.shape[:2]):
            raise ValueError(
                f"iou_scores must have shape {tuple(self.low_resolution_logits.shape[:2])}. "
                + f"got {tuple(self.iou_scores.shape)}"
            )

    @property
    def masks_per_prompt(self) -> int:
        """
        Number of masks predicted per prompt.

        Returns
        -------
        int
            K.
        """
        return int(self.low_resolution_logits.shape[1])

    def best(self) -> "DecodedMasks":
        """
        Keep the mask with the highest predicted IoU of every prompt.

        Returns
        -------
        DecodedMasks
            One mask per prompt, shape (N, 1, h, w).
        """
        best_indices: torch.Tensor = self.iou_scores.argmax(dim=1)
        prompt_indices: torch.Tensor = torch.arange(self.iou_scores.shape[0])
        return DecodedMasks(
            low_resolution_logits=self.low_resolution_logits[prompt_indices, best_indices][:, None],
            iou_scores=self.iou_scores[prompt_indices, best_indices][:, None],
        )
