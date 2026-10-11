# image

## Overview

Images accepted by every segmenter, mask generator and tracker: PIL images of any mode or
[ImageContainer](https://github.com/ry-yoshida-dev/ImageContainer) containers of any channel order (array, PIL,
tensor; RGB, BGR, gray ...). Models read RGB PIL images; 16-bit, 32-bit and float grayscale images are stretched
from their minimum to their maximum onto 8 bits instead of being clipped by PIL.

## Components

| Component | Description |
| --------- | ----------- |
| [image_input.py](./image_input.py) | `ImageInput`: type alias of the accepted image types. |
| [rgb_image.py](./rgb_image.py) | `RgbImage`: converts an `ImageInput` into the RGB PIL image a model reads. |

## Examples

```python
rgb = RgbImage.from_input(ImageContainer.register(cv2.imread("frame.png"), ChannelOrder.BGR))
```
