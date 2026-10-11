# sam2

## Overview

SAM 2 and SAM 2.1 through `transformers.Sam2Model` (`PromptableBackend.SAM2`). The image is stretched over the
1024x1024 input and the embedding is multi-scale. The published checkpoints are video checkpoints (`model_type`
`sam2_video`); their image part is loaded here, and the video predictor lives in
[memory_tracker/](../memory_tracker/README.md). EdgeTAM reuses this segmenter with its own model
([edgetam/](../edgetam/README.md)).

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `Sam2PromptableSegmenter`. |
