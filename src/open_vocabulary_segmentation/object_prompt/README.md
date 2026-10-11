# object_prompt

## Overview

What a promptable model or video tracker segments: one `ObjectPrompt` marks one object with any combination of
positive points (on the object), negative points (off it), one box and one mask, as
[Geometry](https://github.com/ry-yoshida-dev/Geometry) `Points2D` and `Box2D` (any box format) in image pixels. The
mask is either a `BinaryImage` of the image or the `MaskLogits` of a previous prediction, which is how SAM refines a
mask click by click. Something must mark the object: positive points, a box or a mask.

Prompts are validated against the image before inference (coordinates inside the image, mask of the image size) and
against the model (elements it accepts, logits of the same checkpoint and image).

## Components

| Component | Description |
| --------- | ----------- |
| [object_prompt.py](./object_prompt.py) | `ObjectPrompt`: points, box and mask of one object; plain-coordinate constructor, added clicks, validation. |
| [mask_logits.py](./mask_logits.py) | `MaskLogits`: low-resolution logits of a prediction, tied to its checkpoint and image size. |
| [mask_prompt.py](./mask_prompt.py) | `MaskPrompt`: type alias of `MaskLogits` and `BinaryImage`. |

## Examples

```python
prompt = ObjectPrompt.from_coordinates(positive_points=[(420, 310)], negative_points=[(600, 300)])
boxed = ObjectPrompt(box=Box2D.register(value=np.array([50, 60, 250, 340]), box2d_format=Box2DFormat.XYWH))
from_mask = ObjectPrompt(mask=BinaryImage(value=mask))

result = segmenter.segment(encoding, [prompt])
next_prompt = result[0].refine(negative_points=Points2D(value=np.array([[380.0, 290.0]])))
```
