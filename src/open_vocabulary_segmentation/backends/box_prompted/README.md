# box_prompted

## Overview

Open-vocabulary segmentation of backends that only mask given boxes (Grounded-SAM style): SAM and SAM 2. The box
detector built from `SegmenterSettings.box_detector` finds the instances of the prompt, so accepted query kinds,
confidences and matched queries are those of the detector, and its thresholds and NMS select the boxes. The
promptable segmenter of the backend (`mask_segmenter`) masks them: boxes are clipped to the image, the images of a
mini-batch with at least one box are encoded together, and all boxes of an image are decoded in shared passes with
the single-mask head. Images without boxes skip the segmentation model.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `BoxPromptedSegmenter`: box detection and masking through a `PromptableSegmenter`. |
