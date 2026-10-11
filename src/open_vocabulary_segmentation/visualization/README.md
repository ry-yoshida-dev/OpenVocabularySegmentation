# visualization

## Overview

Rendering of every result type over its source image. The source image is any
[ImageContainer](https://github.com/ry-yoshida-dev/ImageContainer) container, and the rendering is returned as an RGB
`ArrayImageContainer`, ready to `save` or convert.

| Result | Method | Color by |
| ------ | ------ | -------- |
| `SegmentationResult` | `render` | Class, most confident on top |
| `ObjectSegmentationResult` | `render_objects` | Prompt, with points and boxes drawn |
| `TrackedFrame` | `render_tracked` | Object id |
| `ConceptFrame` | `render_concepts` | Track id |
| `MaskGenerationResult` | `render_generated` | Region, large regions first |
| Any masks | `render_masks` | Given color ids and drawing order |

## Components

| Component | Description |
| --------- | ----------- |
| [overlay_renderer.py](./overlay_renderer.py) | `MaskOverlayRenderer`: tints and outlines masks in palette colors and draws prompt points and boxes. |

## Examples

```python
renderer = MaskOverlayRenderer(opacity=0.5, contour_thickness=2)
renderer.render(source, result.filter_by_classes(("cat", "dog"))).save("overlay.png")
renderer.render_objects(source, object_result).save("clicks.png")
renderer.render_tracked(frame_container, tracked_frame).save("frame.png")
```
