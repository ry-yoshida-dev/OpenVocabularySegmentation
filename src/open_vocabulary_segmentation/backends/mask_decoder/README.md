# mask_decoder

## Overview

Shared implementation of SAM-family promptable segmenters (SAM, SAM 2, SAM 3 tracker, EdgeTAM): an image encoder run once per
image, and a prompt encoder plus mask decoder run per prompt pass. `MaskDecoderSegmenter` groups the prompts of an
image by `PromptSignature` (point count, box, mask) so every pass decodes prompts of one shape without padding, in
passes of `decode_batch_size`; prompts with a mask are decoded one at a time, since SAM-family decoders add the dense
mask embedding before repeating the image embedding per prompt. `MaskOutput` selects the one-mask or three-mask head
per group. Logits are upscaled a few masks at a time, binarized at 0 and scored for stability (IoU of the masks at
±1 around the threshold); predicted IoU is clipped to `[0, 1]`.

Points, boxes and binary masks are mapped into the model input frame by `ModelFrame`; a binary mask becomes logits of
±10 at the mask prompt resolution (4x the main embedding side: 256 for SAM and SAM 2, 288 for SAM 3).

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `MaskDecoderSegmenter`: grouping, passes, head selection, upscaling and stability scores; subclasses add the model calls. |
| [prompt_signature.py](./prompt_signature.py) | `PromptSignature`: shape of a prompt, whether it is ambiguous and batchable. |
| [prompt_tensors.py](./prompt_tensors.py) | `PromptTensors`: prompts of one signature as point, label, box and mask tensors in the model frame. |
| [point_label.py](./point_label.py) | `PointLabel`: positive, negative and not-a-point labels of SAM prompt encoders. |
| [decoded_masks.py](./decoded_masks.py) | `DecodedMasks`: logits and predicted IoU of one pass, with best-candidate selection. |
| [preprocessed_images.py](./preprocessed_images.py) | `PreprocessedImages`: pixel values and frames of a mini-batch. |
| [prompt_mask.py](./prompt_mask.py) | `PromptMask`: one decoded mask before the masks of an image are assembled. |
