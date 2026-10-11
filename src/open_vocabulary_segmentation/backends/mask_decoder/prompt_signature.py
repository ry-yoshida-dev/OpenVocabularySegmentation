from dataclasses import dataclass

from ...object_prompt import ObjectPrompt


@dataclass(frozen=True)
class PromptSignature:
    """
    Shape of an object prompt; prompts of one signature are decoded together without padding.

    Attributes
    ----------
    point_count : int
        Number of points, positive and negative.
    is_box_given : bool
        Whether the prompt has a box.
    is_mask_given : bool
        Whether the prompt has a mask.
    """

    point_count: int
    is_box_given: bool
    is_mask_given: bool

    @classmethod
    def of(cls, prompt: ObjectPrompt) -> "PromptSignature":
        """
        Signature of a prompt.

        Parameters
        ----------
        prompt : ObjectPrompt
            Object prompt.

        Returns
        -------
        PromptSignature
            Point count and presence of box and mask.
        """
        return cls(
            point_count=prompt.point_count, is_box_given=prompt.box is not None, is_mask_given=prompt.mask is not None
        )

    @property
    def is_ambiguous(self) -> bool:
        """
        Whether the prompts are single points without box or mask.

        Returns
        -------
        bool
            True for one point only.
        """
        return self.point_count == 1 and not self.is_box_given and not self.is_mask_given

    @property
    def is_batchable(self) -> bool:
        """
        Whether several prompts of the signature can share one decoder pass.

        SAM-family decoders add a dense mask embedding to the image embedding before repeating it per prompt, which
        only works for one mask prompt per pass.

        Returns
        -------
        bool
            False for prompts with a mask.
        """
        return not self.is_mask_given
