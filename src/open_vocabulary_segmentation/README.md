# open_vocabulary_segmentation

## Overview

Model-independent interfaces, settings and result types for open-vocabulary, promptable and video segmentation, the
YAML presets of every supported model, and the backends implementing the interfaces. Prompts of classes, devices and
detector settings are the types of `open_vocabulary_detector`, re-exported here. Images may be given as PIL images or
as `image_container` containers.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `OpenVocabularySegmenter`: runtime-independent base with `segment`, `segment_images`, optional box exemplars, prompt kind validation and shared post-processing. |
| [settings.py](./settings.py) | `SegmenterSettings`: backend, weights, thresholds, box detector, batch size, device and precision; `build()` loads the segmenter. |
| [promptable/](./promptable/README.md) | `PromptableSegmenter`: segments the object of each point, box or mask prompt, reusing image encodings. |
| [object_prompt/](./object_prompt/README.md) | `ObjectPrompt` and `MaskLogits`: points, box and mask marking one object. |
| [exemplars/](./exemplars/README.md) | `BoxExemplars` and `ImageExemplars`: SAM 3 box exemplars of a concept. |
| [generation/](./generation/README.md) | `AutomaticMaskGenerator`: segments everything with a point grid over crops. |
| [video/](./video/README.md) | `ObjectTracker` and `ConceptTracker`: offline and streaming video tracking. |
| [options/](./options/README.md) | Backends, precision, mask output, prompt elements, propagation direction and thresholds. |
| [result/](./result/README.md) | Result types of every interface and the mask tools they share. |
| [visualization/](./visualization/README.md) | `MaskOverlayRenderer`: draws every result type over its source image. |
| [image/](./image/README.md) | `ImageInput` and `RgbImage`: accepted image types and their conversion. |
| [runtime/](./runtime/README.md) | `TorchRuntime`: device, precision and model preparation for PyTorch-backed models. |
| [config/](./config/README.md) | YAML presets of every backend and interface. |
| [backends/](./backends/README.md) | Implementations of every interface for SAM, SAM 2, SAM 3, EdgeTAM and Florence-2. |
| [array_types.py](./array_types.py) | NumPy array aliases (`FloatArray`, `Float32Array`, `IntArray`, `BoolArray`). |

## Examples

```python
segmenter = settings.build()
result = segmenter.segment(image, Prompt.from_class_names(("cat", "dog")))

promptable = PromptableSegmenterSettings(
    backend=PromptableBackend.SAM2, weights_path="facebook/sam2.1-hiera-tiny"
).build()
objects = promptable.segment(image, [ObjectPrompt.from_coordinates(positive_points=[(420, 310)])])
```
