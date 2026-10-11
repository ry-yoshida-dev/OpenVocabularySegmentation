from transformers.models.sam3.modeling_sam3 import Sam3ImageSegmentationOutput, Sam3VisionEncoderOutput
from transformers.models.sam3_lite_text.modeling_sam3_lite_text import (
    Sam3LiteTextImageSegmentationOutput,
    Sam3LiteTextVisionEncoderOutput,
)

type Sam3VisionOutput = Sam3VisionEncoderOutput | Sam3LiteTextVisionEncoderOutput
type Sam3SegmentationOutput = Sam3ImageSegmentationOutput | Sam3LiteTextImageSegmentationOutput
