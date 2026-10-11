# result

## Overview

Model-independent segmentation output: instances of open-vocabulary segmenters (`SegmentationResult`), masks of
object prompts (`ObjectSegmentationResult`), regions of automatic mask generation (`MaskGenerationResult`) and class
probabilities of SAM 3's semantic head (`SemanticSegmentationResult`). Masks are boolean arrays at the resolution of the
source image; boxes are derived from the masks and exposed as `geometry.Boxes2D` in absolute XYXY pixel coordinates.
Masks combined into one image (per class, foreground) are returned as
[ImageContainer](https://github.com/ry-yoshida-dev/ImageContainer) `BinaryImage`s, and the source image is passed as
any `ImageContainer` (array, PIL or tensor, any channel order) to crop or cut instances out of it.

Mask NMS counts overlapping pixels only inside the intersection of two boxes, so its cost follows the overlap, not
the image size.

| Need | `SegmentationResult` | `Segmentation` |
| ---- | -------------------- | -------------- |
| Select instances | `filter_by_class(es)`, `filter_by_query`, `filter_by_confidence`, `filter_by_area`, `top_k`, `select` | - |
| Merge duplicates | `non_maximum_suppression` (class-wise or class-agnostic) | - |
| Masks | `class_mask`, `class_masks`, `union_mask` (`BinaryImage`) | `mask`, `binary_image` |
| Label maps | `class_map` (class id per pixel), `instance_map` (instance index per pixel) | - |
| Statistics | `class_counts`, `areas`, `boxes` | `area`, `box` |
| Source image | - | `crop`, `cutout`, `crop_slice` |
| Export | - | `to_polygons` |

Object prompts may yield empty masks, so `ObjectSegmentationResult` keeps them (with `None` boxes) to stay aligned
with its prompts; `prompt_indices` maps each mask to its prompt, three per prompt with `MaskOutput.ALL_CANDIDATES`.

## Components

| Component | Description |
| --------- | ----------- |
| [object_segmentation_result.py](./object_segmentation_result.py) | `ObjectSegmentationResult`: masks, predicted IoU, stability and logits per object prompt; filters, best candidate per prompt, NMS, label map. |
| [object_segmentation.py](./object_segmentation.py) | `ObjectSegmentation`: one mask of one prompt; `refine` builds the next prompt from corrective clicks and its logits. |
| [mask_generation_result.py](./mask_generation_result.py) | `MaskGenerationResult`: regions of automatic mask generation with scores, prompting points and crops. |
| [generated_mask.py](./generated_mask.py) | `GeneratedMask`: one generated region. |
| [semantic_segmentation_result.py](./semantic_segmentation_result.py) | `SemanticSegmentationResult`: per-class probability maps, class masks and class map. |
| [mask_bounds.py](./mask_bounds.py) | `MaskBounds`: pixel bounding boxes of masks. |
| [mask_suppression.py](./mask_suppression.py) | `MaskSuppression`: greedy mask NMS, optionally per group. |
| [segmentation_result.py](./segmentation_result.py) | `SegmentationResult`: masks, confidences and matched query ids for one image with their `Prompt`; selection, mask NMS, class and union masks, label maps and counts. |
| [segmentation.py](./segmentation.py) | `Segmentation`: one instance (mask, `geometry.Box2D`, confidence, class, matched query), yielded by iterating a result; crops, cutouts and polygons. |
| [mask_overlap.py](./mask_overlap.py) | `MaskOverlap`: pairwise mask IoU restricted to box intersections. |
| [mask_contour.py](./mask_contour.py) | `MaskContour`: outer boundaries of a mask as polygons. |
| [source_image.py](./source_image.py) | `SourceImage`: checks that an `ImageContainer` matches the mask size and reads it as RGB. |

## Examples

```python
result = segmenter.segment(image, Prompt.from_texts({"car": ("car", "suv"), "person": ("person",)}))

person_mask = result.class_mask("person")  # BinaryImage, union of every person
person_mask.connected_components()
counts = result.class_counts()  # {"car": 3, "person": 1}
large_cars = result.filter_by_class("car").filter_by_area(500).top_k(5)
class_map = result.class_map()  # class id per pixel, -1 for background

source = ImageContainer.register(image, ChannelOrder.RGB)
for segmentation in large_cars:
    segmentation.cutout(source, background_value=255).save(f"car_{segmentation.area}.png")
    polygons = segmentation.to_polygons()

objects = promptable_segmenter.segment(encoding, prompts, MaskOutput.ALL_CANDIDATES).best_per_prompt()
stable = objects.filter_by_stability(0.9).filter_by_area(1)
```
