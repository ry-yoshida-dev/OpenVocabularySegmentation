# sam

## Overview

Segment Anything through `transformers.SamModel` (`SegmenterBackend.SAM`), masking the boxes of the box detector
with one mask per box. Checkpoints must have `model_type` `sam`; SAM-HQ checkpoints are rejected because
`transformers` returns corrupted masks for them.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `SamSegmenter`. |
