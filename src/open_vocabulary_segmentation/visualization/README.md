# visualization

## Overview

Rendering of segmentation results over their source image. The source image is any
[ImageContainer](https://github.com/ry-yoshida-dev/ImageContainer) container, and the rendering is returned as an RGB
`ArrayImageContainer`, ready to `save` or convert.

## Components

| Component | Description |
| --------- | ----------- |
| [overlay_renderer.py](./overlay_renderer.py) | `MaskOverlayRenderer`: tints every instance with its class color and outlines it, most confident instance on top. |

## Examples

```python
renderer = MaskOverlayRenderer(opacity=0.5, contour_thickness=2)
renderer.render(source, result.filter_by_classes(("cat", "dog"))).save("overlay.png")
```
