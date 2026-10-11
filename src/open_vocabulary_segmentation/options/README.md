# options

## Overview

Enumerations and small option types composing the settings of segmenters, promptable segmenters and trackers, shared
by every backend. The device is `open_vocabulary_detector.Device`.

## Components

| Component | Description |
| --------- | ----------- |
| [backend.py](./backend.py) | `SegmenterBackend`: open-vocabulary segmenter family, whether it masks detector boxes and with which promptable family. |
| [promptable_backend.py](./promptable_backend.py) | `PromptableBackend`: promptable family, its checkpoint model types, prompt elements and mask-only support. |
| [tracker_backend.py](./tracker_backend.py) | `TrackerBackend`: video object tracker family and its checkpoint model types. |
| [prompt_element.py](./prompt_element.py) | `PromptElement`: point, box or mask. |
| [mask_output.py](./mask_output.py) | `MaskOutput`: which masks of the one-mask and three-mask decoder heads are returned. |
| [propagation_direction.py](./propagation_direction.py) | `PropagationDirection`: forward, backward or bidirectional video propagation. |
| [precision.py](./precision.py) | `Precision`: float32, float16 (GPU only) or bfloat16. |
| [thresholds.py](./thresholds.py) | `SegmentationThresholds`: confidence threshold and optional mask NMS IoU threshold. |

## Examples

```python
SegmenterBackend.SAM2.promptable_backend  # PromptableBackend.SAM2
PromptableBackend.FLORENCE2.supported_prompt_elements  # frozenset({PromptElement.BOX})
MaskOutput.AUTO.is_multimask(is_ambiguous=True)  # True
SegmentationThresholds(confidence_threshold=0.5, nms_iou_threshold=0.5)
```
