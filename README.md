# OpenVocabularySegmentation

## Overview

`open_vocabulary_segmentation` puts SAM, SAM 2 / SAM 2.1, SAM 3, EdgeTAM and Florence-2 behind a few model-independent
interfaces covering everything these models can do in Hugging Face `transformers`: segmentation by text, by points,
boxes and masks, of everything in an image, and of objects and concepts through videos. Models are swapped by
changing configuration only; every interface has a settings dataclass, normally built from a YAML preset, and
returns one result type for every backend.

| Interface | What it segments | SAM | SAM 2 | SAM 3 | EdgeTAM | Florence-2 |
| --------- | ---------------- | :-: | :---: | :---: | :-----: | :--------: |
| `OpenVocabularySegmenter` | Every instance of the classes of a text (or visual) `Prompt` | with a box detector | with a box detector | yes (text and visual), plus box exemplars and semantic maps | with a box detector | yes, plus referring expressions |
| `PromptableSegmenter` | The object marked by each `ObjectPrompt` (points, box, mask) | yes | yes | yes | yes | boxes |
| `AutomaticMaskGenerator` | Everything, with a point grid over crops | yes | yes | yes | yes | - |
| `ObjectTracker` | Objects prompted on some video frames, through the video | - | yes | yes | yes | - |
| `ConceptTracker` | Every instance of text or visual queries through a video, with track ids | - | - | yes | - | - |

It builds on [OpenVocabularyDetector](https://github.com/ry-yoshida-dev/OpenVocabularyDetector) (class prompts,
devices and box detectors, re-exported here) and expresses geometry with
[Geometry](https://github.com/ry-yoshida-dev/Geometry) (`Points2D`, `Box2D`, `Boxes2D` in any box format) and masks
with [ImageContainer](https://github.com/ry-yoshida-dev/ImageContainer) (`BinaryImage`).

- **Configuration is data**: weights, thresholds, batch sizes, precision, box detectors and mask generation options
  live in YAML presets under [config/](src/open_vocabulary_segmentation/config/README.md), one folder per interface and
  backend.
- **Prompts of every kind**: classes queried by text phrases and reference images (`Prompt`); objects marked by
  positive and negative points, a box and a mask (`ObjectPrompt`), refined click by click with the logits of the
  previous mask; SAM 3 concepts refined by positive and negative box exemplars or defined by boxes alone
  (`ImageExemplars`). Prompts are validated against the image and the model before any inference.
- **Interactive and fast**: an image is encoded once and prompted any number of times; prompts of one shape are decoded
  together; SAM's ambiguity handling (best of three masks for a single click) is the default, and every candidate is
  available.
- **Video, offline or streaming**: prompt objects on any frame and propagate forward, backward or both, or feed a
  stream frame by frame with bounded memory; SAM 3 video detects and tracks every instance of a concept.
- **Results are comparable and tools**: masks are full-resolution boolean arrays, boxes are derived from masks,
  scores are in `[0, 1]`. Results select, filter, deduplicate (mask NMS), build class, instance, object and track
  maps, crop, cut out and export polygons; `MaskOverlayRenderer` draws every result type and the prompts.
- **Images in, images out**: PIL images or `ImageContainer` containers of any channel order (array, PIL, tensor; RGB,
  BGR, gray ...).

| Backend | Models | Presets |
| ------- | ------ | ------- |
| `sam` | Segment Anything | `vit_base`, `vit_large`, `vit_huge` |
| `sam2` | SAM 2.1 | `hiera_tiny`, `hiera_small`, `hiera_base_plus`, `hiera_large` |
| `sam3` | SAM 3 (detector, tracker, video); SAM 3-LiteText (detector) | `original`; `litetext_s0`, `litetext_s1`, `litetext_l` |
| `edgetam` | EdgeTAM (on-device SAM 2) | `original` |
| `florence2`, `florence2_referring` | Florence-2 | `base`, `base_ft`, `large`, `large_ft` |

SAM, SAM 2 and EdgeTAM only mask given boxes, so as `OpenVocabularySegmenter` they take the boxes of any
`open_vocabulary_detector` model (Grounded-SAM style, e.g. Grounding DINO + SAM 2.1, OWLv2 + SAM); visual queries
work when the detector supports them (OWL-ViT, OWLv2, YOLOE). SAM 3 reads a visual query as the text `"visual"`
plus the geometry tokens of its reference boxes, encoded on the reference images, both on images and in video. SAM 3
scores each instance by its match with the
concept and the presence of the concept in the image. Florence-2 is generative and has no scores (confidence `1.0`,
no mask logits), and a phrase absent from the image usually still gets an instance.

SAM 3 weights (`facebook/sam3`) are gated on the Hugging Face Hub: request access and log in (`hf auth login`) before
first use; one checkpoint serves the detector, the tracker and video. SAM 3-LiteText checkpoints (`vil-uob/sam3-litetext-*`) replace the text encoder with a light one and serve
concept segmentation of images. SAM 3.1 is not supported, since
`facebook/sam3.1` ships only weights that `transformers` cannot load. SAM-HQ is not supported: `transformers` (5.19)
returns corrupted masks for the published SAM-HQ checkpoints.

Masks are kept at the full resolution of the source image, so memory grows with image size times mask count (a 4K
image holds about 8 MB per mask); filter or crop results early. A preprocessed video frame takes about 12 MB, so
offline video sessions keep frames on the CPU by default; streaming sessions keep only what the model reads.

For module details, see [src/open_vocabulary_segmentation/README.md](src/open_vocabulary_segmentation/README.md).

## Installation

```bash
pip install -e .
```

SAM 3 video post-processing (hole filling, speck removal, NMS of new tracks) uses the optional `kernels` package;
without it, `transformers` skips that step:

```bash
pip install -e ".[kernels]"
```

EdgeTAM's image encoder is a `timm` model:

```bash
pip install -e ".[edgetam]"
```

The box detectors come from `open-vocabulary-detector`, installed as a dependency; install its optional extras (e.g.
`ultralytics` for YOLOE boxes) when needed. For development:

```bash
pip install -e ".[dev]"
```

Unit tests run offline; `--run-slow` adds tests of every interface on the real checkpoints:

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

Several phrases for one class, merged by mask NMS, then the result tools on an OpenCV (BGR) image:

```python
import cv2
from image_container import ChannelOrder, ImageContainer

from open_vocabulary_segmentation import MaskOverlayRenderer

source = ImageContainer.register(cv2.imread("street.jpg"), ChannelOrder.BGR)
result = segmenter.segment(source, Prompt.from_texts({"car": ("car", "suv", "taxi"), "person": ("person",)}))

person_mask = result.class_mask("person")  # BinaryImage: union of every person
counts = result.class_counts()  # {"car": 3, "person": 2}
large_cars = result.filter_by_class("car").filter_by_area(500).top_k(3)
for index, car in enumerate(large_cars):
    car.cutout(source, background_value=255).save(f"car_{index}.png")
MaskOverlayRenderer().render(source, result).save("overlay.png")
```

SAM 3 concepts refined by box exemplars, defined by a box alone, and as semantic probability maps:

```python
import numpy as np
from geometry import Box2DFormat, Boxes2D

from open_vocabulary_segmentation import (
    BoxExemplars,
    ImageExemplars,
    Precision,
    SegmentationThresholds,
    SegmenterBackend,
)


def boxes(xyxy):
    return Boxes2D.register(value=np.array(xyxy, dtype=np.float64), box2d_format=Box2DFormat.XYXY)


sam3 = SegmenterSettings(
    backend=SegmenterBackend.SAM3,
    weights_path="facebook/sam3",
    thresholds=SegmentationThresholds(confidence_threshold=0.5, nms_iou_threshold=0.5),
    precision=Precision.BFLOAT16,
).build()
prompt = Prompt.from_texts({"car": ("taxi",)})
exemplars = ImageExemplars(
    {"car": BoxExemplars(positive_boxes=boxes([[30, 40, 180, 160]]), negative_boxes=boxes([[400, 50, 560, 170]]))}
)
taxis = sam3.segment(image, prompt, exemplars)

parts = sam3.segment(
    image,
    Prompt.from_class_names(("part",)),
    ImageExemplars({"part": BoxExemplars(positive_boxes=boxes([[200, 210, 240, 250]]), is_text_ignored=True)}),
)
road = sam3.segment_semantic(image, Prompt.from_class_names(("road", "sidewalk"))).class_map()
```

Interactive segmentation: encode once, click, refine with the previous mask:

```python
from geometry import Points2D

from open_vocabulary_segmentation import MaskOutput, ObjectPrompt, PromptableBackend, PromptableSegmenterSettings

segmenter = PromptableSegmenterSettings(
    backend=PromptableBackend.SAM2, weights_path="facebook/sam2.1-hiera-large"
).build()
encoding = segmenter.encode(image)

result = segmenter.segment(
    encoding,
    [
        ObjectPrompt.from_coordinates(positive_points=[(420, 310)]),
        ObjectPrompt.from_coordinates(box_xyxy=(50, 60, 300, 400), negative_points=[(170, 230)]),
    ],
)
refined = segmenter.segment(encoding, [result[0].refine(negative_points=Points2D(value=np.array([[380.0, 290.0]])))])
candidates = segmenter.segment(
    encoding, [ObjectPrompt.from_coordinates(positive_points=[(420, 310)])], MaskOutput.ALL_CANDIDATES
)
MaskOverlayRenderer().render_objects(source, result).save("clicks.png")
```

Segment everything:

```python
from open_vocabulary_segmentation import MaskGenerationOptions, MaskGeneratorSettings

generator = MaskGeneratorSettings(
    segmenter=PromptableSegmenterSettings(backend=PromptableBackend.SAM2, weights_path="facebook/sam2.1-hiera-large"),
    options=MaskGenerationOptions(points_per_side=32, crop_layer_count=1, minimum_region_area=100),
).build()
regions = generator.generate(image).sort_by_area()
```

Video object tracking, offline in both directions with a correction, or streaming:

```python
from open_vocabulary_segmentation import ObjectTrackerSettings, PropagationDirection, TrackerBackend

tracker = ObjectTrackerSettings(backend=TrackerBackend.SAM2, weights_path="facebook/sam2.1-hiera-tiny").build()

session = tracker.start_session(frames)
session.prompt_objects(
    30,
    {
        1: ObjectPrompt.from_coordinates(positive_points=[(420, 310)]),
        2: ObjectPrompt.from_coordinates(box_xyxy=(50, 60, 300, 400)),
    },
)
tracked = {
    frame.frame_index: frame for frame in session.propagate(PropagationDirection.BIDIRECTIONAL, start_frame_index=30)
}
session.prompt_object(80, 1, ObjectPrompt.from_coordinates(positive_points=[(510, 330)], negative_points=[(560, 300)]))

stream = tracker.start_stream()
for index, frame in enumerate(camera_frames):
    tracked_frame = stream.track_next(
        frame, {1: ObjectPrompt.from_coordinates(positive_points=[(420, 310)])} if index == 0 else None
    )
```

Video concept tracking with SAM 3:

```python
from open_vocabulary_segmentation import ConceptTrackerSettings

tracker = ConceptTrackerSettings(weights_path="facebook/sam3", confidence_threshold=0.5).build()
session = tracker.start_session(frames, Prompt.from_texts({"person": ("person",), "car": ("car", "truck")}))
for frame in session.propagate():
    for tracked in frame:
        print(frame.frame_index, tracked.track_id, tracked.segmentation.class_name, tracked.segmentation.confidence)
```

Command-line examples for every interface are in [examples/](examples/README.md):

```bash
python examples/segment_directory.py images/ overlays/ --backend sam2 --weights facebook/sam2.1-hiera-tiny \
    --box-detector-backend grounding_dino --box-detector-weights IDEA-Research/grounding-dino-tiny \
    --classes cat "car:car,suv,taxi"
python examples/track_video.py clip.mp4 tracked.mp4 --backend sam3 --weights facebook/sam3 --object "1:point=420,310"
```
