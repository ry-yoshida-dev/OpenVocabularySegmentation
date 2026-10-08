# OpenVocabularySegmentation

## Overview

`open_vocabulary_segmentation` puts SAM, SAM 2 / SAM 2.1, SAM 3 and Florence-2 behind one interface so that models can
be swapped and compared by changing configuration only. Application code builds a prompt, calls one
`OpenVocabularySegmenter`, and reads one `SegmentationResult` type of instance masks; which model runs is decided by a
`SegmenterSettings`, normally built from a YAML preset.

It builds on [OpenVocabularyDetector](https://github.com/ry-yoshida-dev/OpenVocabularyDetector): prompts (`Prompt`,
`TextQuery`, `VisualQuery`), devices and detector settings are the same types, re-exported from this package.

- **Configuration is data**: every model-specific value (weights, thresholds, batch size, the box detector) lives in a
  YAML preset under [config/](src/open_vocabulary_segmentation/config/README.md), one folder per backend.
- **Two ways to find instances**: SAM and SAM 2 only mask given boxes, so they take the boxes of any
  `open_vocabulary_detector` model (Grounded-SAM style, e.g. Grounding DINO + SAM 2.1, OWLv2 + SAM). SAM 3 and
  Florence-2 find every instance of a text phrase themselves.
- **Classes are separate from queries**: a `Prompt` maps each class name to its queries, e.g. `"car"` by `"car"`,
  `"suv"` and `"taxi"`. Every instance reports its class together with the query that matched, and class-wise mask
  NMS merges instances of one class found by different queries.
- **Results are comparable**: masks are full-resolution boolean arrays of the source image, boxes are derived from the
  masks as [Geometry](https://github.com/ry-yoshida-dev/Geometry) `Boxes2D` (absolute XYXY pixels), confidences are
  in `[0, 1]`, and every backend applies the confidence threshold, optional mask NMS and sorting in the same way.
- **Results are tools**: `SegmentationResult` selects instances by class, query, area or rank, builds the mask of a
  class or of the foreground as an [ImageContainer](https://github.com/ry-yoshida-dev/ImageContainer) `BinaryImage`,
  class and instance label maps, and counts; each `Segmentation` crops or cuts its instance out of the source image
  and exports polygons. `MaskOverlayRenderer` draws a result over its image.
- **Images in, images out**: segmenters and result tools accept PIL images or `ImageContainer` containers of any
  channel order (array, PIL, tensor; RGB, BGR, gray ...).

| Backend | Models | Finds instances by | Query kinds | Presets |
| ------- | ------ | ------------------ | ----------- | ------- |
| `sam` | Segment Anything | Boxes of `box_detector` | Those of the box detector | `vit_base`, `vit_large`, `vit_huge` |
| `sam2` | SAM 2, SAM 2.1 | Boxes of `box_detector` | Those of the box detector | `hiera_tiny`, `hiera_small`, `hiera_base_plus`, `hiera_large` (SAM 2.1) |
| `sam3` | SAM 3 | Itself (concept segmentation) | `TextQuery` | `original` |
| `florence2` | Florence-2 | Itself (detection, then region segmentation) | `TextQuery` | `base`, `base_ft`, `large`, `large_ft` |

With a box-prompted backend, the box detector decides the accepted query kinds, the confidences and the matched
queries; visual queries work when the detector supports them (OWL-ViT, OWLv2, YOLOE). The box detector's own
thresholds and NMS select the boxes, and each box becomes one mask.

SAM 3 scores each instance by its match with the phrase and the presence of the phrase in the image. Its weights
(`facebook/sam3`) are gated on the Hugging Face Hub: request access and log in (`hf auth login`) before first use.

Florence-2 is a generative model without scores: every instance has confidence `1.0`, and it tends to return an
instance even for a phrase that is not in the image. It runs one beam search per text query and one per detected
instance, so it is much slower than the other backends.

Masks are kept at the full resolution of the source image, one boolean `(H, W)` array per instance, so memory grows
with image size times instance count (a 4K image holds about 8 MB per instance); filter or crop results early when
segmenting large images with many instances.

SAM-HQ is not supported: `transformers` (5.19) returns corrupted masks for the published SAM-HQ checkpoints.

For module details, see [src/open_vocabulary_segmentation/README.md](src/open_vocabulary_segmentation/README.md).

## Installation

```bash
pip install -e .
```

The box detectors come from `open-vocabulary-detector`, installed as a dependency; install its optional extras (e.g.
`ultralytics` for YOLOE boxes) when needed. For development:

```bash
pip install -e ".[dev]"
```

Unit tests run offline; `--run-slow` adds smoke tests that load the real checkpoints of every backend:

```bash
pytest
pytest --run-slow
```

## Examples

Build settings from a preset with [DictConfigHandler](https://github.com/ry-yoshida-dev/DictConfigHandler) (not a
dependency of this package), then run any model with the same code.

```python
from dictconfig_handler import DictConfigHandler
from omegaconf import OmegaConf

from open_vocabulary_segmentation import Prompt, SegmenterSettings

preset_path = "src/open_vocabulary_segmentation/config/sam2/hiera_tiny.yaml"
settings = DictConfigHandler(cfg=OmegaConf.load(preset_path)).build_dataclass(SegmenterSettings, key="segmenter")
segmenter = settings.build()

results = segmenter.segment_images(images, Prompt.from_class_names(("cat", "remote control")))
for segmentation in results[0]:
    print(segmentation.class_name, segmentation.confidence, segmentation.area, segmentation.box.value)
```

Several phrases for one class, merged by mask NMS (`nms_iou_threshold`), then the result tools on an OpenCV (BGR)
image:

```python
import cv2
from image_container import ChannelOrder, ImageContainer

from open_vocabulary_segmentation import MaskOverlayRenderer

source = ImageContainer.register(cv2.imread("street.jpg"), ChannelOrder.BGR)
result = segmenter.segment(source, Prompt.from_texts({"car": ("car", "suv", "taxi"), "person": ("person",)}))

person_mask = result.class_mask("person")               # BinaryImage: union of every person
counts = result.class_counts()                          # {"car": 3, "person": 2}
class_map = result.class_map()                          # class id per pixel, -1 for background
large_cars = result.filter_by_class("car").filter_by_area(500).top_k(3)
for index, car in enumerate(large_cars):
    car.cutout(source, background_value=255).save(f"car_{index}.png")
    polygons = car.to_polygons()

MaskOverlayRenderer().render(source, result).save("overlay.png")
```

Settings in code, with OWLv2 finding the boxes SAM masks:

```python
from open_vocabulary_segmentation import (
    DetectionThresholds,
    DetectorBackend,
    DetectorSettings,
    SegmentationThresholds,
    SegmenterBackend,
    SegmenterSettings,
)

settings = SegmenterSettings(
    backend=SegmenterBackend.SAM,
    weights_path="facebook/sam-vit-base",
    thresholds=SegmentationThresholds(confidence_threshold=0.0, nms_iou_threshold=None),
    box_detector=DetectorSettings(
        backend=DetectorBackend.OWL_VIT,
        weights_path="google/owlv2-base-patch16-ensemble",
        thresholds=DetectionThresholds(confidence_threshold=0.2, nms_iou_threshold=0.3),
    ),
)
```

Command-line example over a directory, saving mask overlays:

```bash
python examples/segment_directory.py images/ overlays/ --backend sam2 --weights facebook/sam2.1-hiera-tiny \
    --box-detector-backend grounding_dino --box-detector-weights IDEA-Research/grounding-dino-tiny \
    --classes cat "car:car,suv,taxi"
```
