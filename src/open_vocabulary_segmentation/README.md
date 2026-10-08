# open_vocabulary_segmentation

## Overview

Model-independent interface, settings and result types for open-vocabulary instance segmentation, the YAML presets of
every supported model, and the backends implementing the interface. Prompts, devices and detector settings are the
types of `open_vocabulary_detector`, re-exported here. Images may be given as PIL images or as `image_container`
containers.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `OpenVocabularySegmenter`: runtime-independent base with `segment`, `segment_images`, prompt kind validation and shared post-processing. |
| [settings.py](./settings.py) | `SegmenterSettings`: backend, weights, thresholds, box detector, batch size, device and precision; `build()` loads the segmenter. |
| [options/](./options/README.md) | `SegmenterBackend` and `SegmentationThresholds`: options composing `SegmenterSettings`. |
| [result/](./result/README.md) | `SegmentationResult` and `Segmentation`: selection, mask NMS, class masks, label maps, crops, cutouts and polygons. |
| [visualization/](./visualization/README.md) | `MaskOverlayRenderer`: draws instances over their source image. |
| [runtime/](./runtime/README.md) | `TorchRuntime`: device, precision and model preparation for PyTorch-backed segmenters. |
| [config/](./config/README.md) | YAML presets of every backend, buildable into `SegmenterSettings`. |
| [backends/](./backends/README.md) | SAM and SAM 2 (box-prompted), SAM 3 and Florence-2 segmenters. |
| [array_types.py](./array_types.py) | NumPy array aliases (`FloatArray`, `IntArray`, `BoolArray`). |

## Examples

```python
segmenter = settings.build()

result = segmenter.segment(image, Prompt.from_class_names(("cat", "dog")))
results = segmenter.segment_images(images, Prompt.from_texts({"car": ("car", "suv", "taxi")}))
```
