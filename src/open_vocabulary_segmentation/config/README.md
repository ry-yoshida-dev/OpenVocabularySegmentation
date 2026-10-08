# config

## Overview

Ready-made YAML presets of every supported model, one folder per `SegmenterBackend` value, named by model variant
and size.

Each preset's `segmenter` section maps one-to-one onto the fields of [`SegmenterSettings`](../settings.py) (enums
written by value, `thresholds` as a nested section, `null` disabling NMS), so it can be built with any dataclass
builder such as `DictConfigHandler.build_dataclass(SegmenterSettings, key="segmenter")`. Box-prompted presets
(`sam`, `sam2`) also carry a `box_detector` section in the format of `open_vocabulary_detector.DetectorSettings`,
using Grounding DINO tiny; replace it with any detector preset of that package.

| Folder | Presets |
| ------ | ------- |
| [sam/](./sam) | SAM `vit_base`, `vit_large`, `vit_huge` + Grounding DINO tiny |
| [sam2/](./sam2) | SAM 2.1 `hiera_tiny`, `hiera_small`, `hiera_base_plus`, `hiera_large` + Grounding DINO tiny |
| [sam3/](./sam3) | SAM 3 `original` (gated weights) |
| [florence2/](./florence2) | `base`, `base_ft`, `large`, `large_ft` (confidence threshold `0.0`, since every instance has confidence `1.0`) |

## Examples

```yaml
segmenter:
  backend: sam2
  weights_path: facebook/sam2.1-hiera-tiny
  batch_size: 4
  device: auto
  is_half_precision_enabled: false
  thresholds:
    confidence_threshold: 0.0
    nms_iou_threshold: null
  box_detector:
    backend: grounding_dino
    weights_path: IDEA-Research/grounding-dino-tiny
    batch_size: 4
    device: auto
    is_half_precision_enabled: false
    thresholds:
      confidence_threshold: 0.35
      nms_iou_threshold: 0.5
```

For box-prompted presets, the box detector's thresholds select the boxes and the segmenter thresholds then apply to
the masks; a preset is a starting point to copy into an application config.
