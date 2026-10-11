# sam

## Overview

Segment Anything through `transformers.SamModel` (`PromptableBackend.SAM`). SAM resizes the longest image side to
1024 and pads the rest, so prompts are scaled by the resize factor and masks are cropped back from the padded frame.
Checkpoints must have `model_type` `sam`; SAM-HQ checkpoints are rejected because `transformers` returns corrupted
masks for them. SAM was trained with mask prompts only next to points, so a mask alone is rejected.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `SamPromptableSegmenter`. |
