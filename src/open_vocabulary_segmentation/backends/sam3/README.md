# sam3

## Overview

SAM 3 through `transformers.Sam3Model` (`SegmenterBackend.SAM3`), using promptable concept segmentation: the model
finds and masks every instance of a text phrase. The images of a mini-batch are encoded once, the text queries of a
prompt are encoded once and reused while the same prompt is segmented, and the detection and mask decoders run once
per query over the mini-batch. An instance scores its query match times the presence of the phrase in the image;
masks are binarized at `MASK_THRESHOLD` (0.5).

The `facebook/sam3` weights are gated on the Hugging Face Hub.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `Sam3Segmenter`. |
| [prompt_encoding.py](./prompt_encoding.py) | `PromptEncoding`: text features of every query of a prompt, reused across image batches. |
