from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..options import PromptElement
from ..promptable import PromptableSegmenterSettings
from .options import MaskGenerationOptions

if TYPE_CHECKING:
    from .generator import AutomaticMaskGenerator


@dataclass(frozen=True, kw_only=True)
class MaskGeneratorSettings:
    """
    Promptable segmenter and options of automatic mask generation.

    The fields map one-to-one onto the ``mask_generator`` section of the YAML presets under ``config/generation``.

    Attributes
    ----------
    segmenter : PromptableSegmenterSettings
        Promptable segmenter prompted with the point grids.
    options : MaskGenerationOptions
        Grid, crop and filtering parameters.
    """

    segmenter: PromptableSegmenterSettings
    options: MaskGenerationOptions = field(default_factory=MaskGenerationOptions)

    def build(self) -> "AutomaticMaskGenerator":
        """
        Load the segmenter and wrap it in a mask generator.

        Raises
        ------
        ValueError
            If the backend does not accept point prompts.

        Returns
        -------
        AutomaticMaskGenerator
            Ready mask generator.
        """
        from .generator import AutomaticMaskGenerator

        if PromptElement.POINT not in self.segmenter.backend.supported_prompt_elements:
            raise ValueError(f"mask generation needs point prompts, which {self.segmenter.backend} does not accept.")

        return AutomaticMaskGenerator(self.segmenter.build(), self.options)
