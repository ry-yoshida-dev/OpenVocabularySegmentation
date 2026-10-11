# promptable

## Overview

Promptable visual segmentation: segment the single object each `ObjectPrompt` marks. Backends are SAM, SAM 2, EdgeTAM,
the SAM 3 tracker (points, boxes, masks) and Florence-2 (boxes), selected by `PromptableSegmenterSettings`.

An image is encoded once (`encode`) and prompted any number of times, which makes click-by-click refinement
interactive; `segment` also takes raw images. `MaskOutput` picks the decoder head: the best of three masks for
ambiguous single points (default), always one, always the best of three, or all three candidates. Every mask comes
with the predicted IoU, a stability score and the low-resolution logits that refine it in the next step.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `PromptableSegmenter`: runtime-independent base with `encode`, `segment`, mini-batching and prompt validation. |
| [settings.py](./settings.py) | `PromptableSegmenterSettings`: backend, weights, batch sizes, device and precision; `build()` loads the segmenter. |
| [encoded_image.py](./encoded_image.py) | `EncodedImage`: image embedding reusable for any number of prompts of the same segmenter. |
| [model_frame.py](./model_frame.py) | `ModelFrame`: placement of the image in the square model input; maps points, boxes and binary masks into it. |
| [promptable_input.py](./promptable_input.py) | `PromptableInput`: type alias of a raw image or an `EncodedImage`. |

## Examples

```python
segmenter = PromptableSegmenterSettings(
    backend=PromptableBackend.SAM2, weights_path="facebook/sam2.1-hiera-tiny"
).build()
encoding = segmenter.encode(image)
result = segmenter.segment(encoding, [ObjectPrompt.from_coordinates(positive_points=[(420, 310)])])
candidates = segmenter.segment(encoding, [prompt], MaskOutput.ALL_CANDIDATES)
refined = segmenter.segment(encoding, [result[0].refine(negative_points=extra_clicks)])
```
