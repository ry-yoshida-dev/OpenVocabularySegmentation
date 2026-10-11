# exemplars

## Overview

Box exemplars of SAM 3 promptable concept segmentation. Positive boxes drawn on the image show what a class is,
negative boxes show look-alikes to exclude; SAM 3 then segments every instance of the concept, not only the boxed
ones. Exemplars refine the text queries of their class, or replace them with `is_text_ignored` ("everything that
looks like this"). They belong to one image, so `OpenVocabularySegmenter.segment_images` takes one `ImageExemplars`
(or `None`) per image; segmenters without exemplar support reject them.

## Components

| Component | Description |
| --------- | ----------- |
| [box_exemplars.py](./box_exemplars.py) | `BoxExemplars`: positive and negative `Boxes2D` of one class, optionally replacing its text. |
| [image_exemplars.py](./image_exemplars.py) | `ImageExemplars`: box exemplars of one image keyed by class name, validated against the prompt and image. |

## Examples

```python
exemplars = ImageExemplars(
    {
        "car": BoxExemplars(positive_boxes=boxes_of_taxis, negative_boxes=boxes_of_police_cars),
    }
)
taxis = sam3.segment(image, Prompt.from_texts({"car": ("taxi",)}), exemplars)

look_alikes = sam3.segment(
    image,
    Prompt.from_class_names(("part",)),
    ImageExemplars({"part": BoxExemplars(positive_boxes=one_part_box, is_text_ignored=True)}),
)
```
