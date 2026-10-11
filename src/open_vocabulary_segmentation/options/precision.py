from enum import StrEnum


class Precision(StrEnum):
    """
    Floating-point precision of a model's weights and floating-point inputs.

    Prompt coordinates are model inputs too: in ``BFLOAT16`` a point near the far edge of a 1024-pixel model input is
    rounded to a few pixels, as in the ``transformers`` reference usage; use ``FLOAT32`` when clicks must be exact.

    Attributes
    ----------
    FLOAT32 : str
        Single precision; runs on every device.
    FLOAT16 : str
        Half precision; GPU only.
    BFLOAT16 : str
        Brain floating point, the precision SAM 2 and SAM 3 were trained in; runs on every device, fastest on GPUs
        with native support.
    """

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    @property
    def is_cpu_supported(self) -> bool:
        """
        Whether models can run in this precision on the CPU.

        Returns
        -------
        bool
            False for ``FLOAT16`` only.
        """
        match self:
            case Precision.FLOAT32 | Precision.BFLOAT16:
                return True
            case Precision.FLOAT16:
                return False
