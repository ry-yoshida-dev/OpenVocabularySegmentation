# edgetam

## Overview

EdgeTAM through `transformers.EdgeTamModel` (`PromptableBackend.EDGETAM`, `SegmenterBackend.EDGETAM` with a box
detector, `TrackerBackend.EDGETAM`): SAM 2 with a light RepViT image encoder (about 9M parameters in all), for
on-device interactive and video segmentation. It shares the processor, multi-scale embedding and mask decoder of
SAM 2, so the image segmenter reuses the SAM 2 implementation; the video predictor lives in
[memory_tracker/](../memory_tracker/README.md). The image encoder is a `timm` model: install the `edgetam` extra.
The `transformers` checkpoint is `yonigozlan/EdgeTAM-hf` (`model_type` `edgetam_video`).

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `EdgeTamPromptableSegmenter`: SAM 2 promptable segmenter loading `EdgeTamModel`. |
