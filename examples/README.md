# examples

## Overview

Runnable scripts using the package, one per interface.

## Components

| Component | Description |
| --------- | ----------- |
| [segment_directory.py](./segment_directory.py) | Segments classes in every image of a directory with any open-vocabulary backend, prints the instances and saves overlays; `name:query,query` gives a class several text queries. |
| [segment_objects.py](./segment_objects.py) | Segments objects marked by points and boxes, optionally refines the first with corrective clicks, and saves the masks with their prompts. |
| [generate_masks.py](./generate_masks.py) | Segments everything in an image with automatic mask generation and saves the regions. |
| [track_video.py](./track_video.py) | Tracks objects prompted on one frame through a video file, offline (bidirectional) or streaming, and writes the tracked video. |
| [track_concepts.py](./track_concepts.py) | Tracks every instance of text phrases through a video file with SAM 3 video and writes the tracks. |

## Examples

```bash
python examples/segment_directory.py images/ overlays/ --backend sam3 --weights facebook/sam3 --classes person "car:car,truck"
python examples/segment_objects.py photo.jpg clicks.png --backend sam2 --weights facebook/sam2.1-hiera-tiny \
    --object "point=420,310 negative=600,300" --object "box=50,60,300,400" --refine "negative=380,290"
python examples/generate_masks.py photo.jpg everything.png --backend sam2 --weights facebook/sam2.1-hiera-large
python examples/track_video.py clip.mp4 tracked.mp4 --backend sam3 --weights facebook/sam3 --frame 30 \
    --object "1:point=420,310" --object "2:box=50,60,300,400"
python examples/track_concepts.py clip.mp4 tracks.mp4 --classes person "car:car,truck" --stream
```
