# sam2

## Overview

SAM 2 and SAM 2.1 through `transformers.Sam2Model` (`SegmenterBackend.SAM2`), masking the boxes of the box detector
with one mask per box. The published checkpoints are video checkpoints (`model_type` `sam2_video`); their image part
is loaded.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `Sam2Segmenter`. |
