# box_prompted

## Overview

Shared base of segmenters that only mask given boxes (Grounded-SAM style). The box detector built from
`SegmenterSettings.box_detector` finds the instances of the prompt, so accepted query kinds, confidences and matched
queries are those of the detector, and its thresholds and NMS select the boxes. Each box becomes one mask.

Per mini-batch, every image with at least one box is preprocessed once together with its boxes, all such images are
embedded in one pass of the image encoder, and the boxes of each image are decoded in chunks of `BOX_CHUNK_SIZE`.
Images without boxes skip the segmentation model. Subclasses implement only the model-specific preprocessing,
embedding, decoding and upscaling.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `BoxPromptedSegmenter`: box detection, batched image embedding and chunked box decoding. |
