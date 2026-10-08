# options

## Overview

Individual options composing `SegmenterSettings`, shared by every backend: which segmenter family runs and how its
output is post-processed. The device is `open_vocabulary_detector.Device`.

## Components

| Component | Description |
| --------- | ----------- |
| [backend.py](./backend.py) | `SegmenterBackend`: segmenter family and whether it masks the boxes of a detector (`is_box_prompted`). |
| [thresholds.py](./thresholds.py) | `SegmentationThresholds`: confidence threshold and optional mask NMS IoU threshold. |

## Examples

```python
SegmenterBackend.SAM2.is_box_prompted  # True
SegmentationThresholds(confidence_threshold=0.5, nms_iou_threshold=0.5)
```
