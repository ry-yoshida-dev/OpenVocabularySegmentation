# backends

## Overview

Concrete implementations of every interface of the package. Each implementation takes the common settings of its
interface, declares the backend it serves, checks the checkpoint's `model_type` before loading
(`CheckpointConfig`) and runs its model through a `TorchRuntime`.

| Interface | SAM | SAM 2 | SAM 3 | EdgeTAM | Florence-2 |
| --------- | --- | ----- | ----- | ------- | ---------- |
| `OpenVocabularySegmenter` | `BoxPromptedSegmenter` | `BoxPromptedSegmenter` | `Sam3Segmenter` | `BoxPromptedSegmenter` | `Florence2Segmenter`, `Florence2ReferringSegmenter` |
| `PromptableSegmenter` | `SamPromptableSegmenter` | `Sam2PromptableSegmenter` | `Sam3PromptableSegmenter` | `EdgeTamPromptableSegmenter` | `Florence2PromptableSegmenter` |
| `AutomaticMaskGenerator` | yes | yes | yes | yes | - |
| `ObjectTracker` | - | `MemoryObjectTracker` | `MemoryObjectTracker` | `MemoryObjectTracker` | - |
| `ConceptTracker` | - | - | `Sam3ConceptTracker` | - | - |

Nothing is imported eagerly: settings import only the sub-package of the requested backend, so importing
`open_vocabulary_segmentation` loads no `transformers` model.

## Components

| Component | Description |
| --------- | ----------- |
| [checkpoint_config.py](./checkpoint_config.py) | `CheckpointConfig`: reads and checks the `model_type` of a `transformers` checkpoint without loading its weights. |
| [mask_decoder/](./mask_decoder/README.md) | `MaskDecoderSegmenter`: shared prompt grouping, decoding and scoring of SAM-family promptable segmenters. |
| [box_prompted/](./box_prompted/README.md) | `BoxPromptedSegmenter`: open-vocabulary detector boxes masked by a promptable segmenter (Grounded-SAM). |
| [sam/](./sam/README.md) | SAM promptable segmenter. |
| [sam2/](./sam2/README.md) | SAM 2 / SAM 2.1 promptable segmenter. |
| [edgetam/](./edgetam/README.md) | EdgeTAM promptable segmenter, reusing the SAM 2 implementation. |
| [sam3/](./sam3/README.md) | SAM 3 concept segmentation (text, exemplars, semantics), promptable tracker and video concept tracker. |
| [memory_tracker/](./memory_tracker/README.md) | Video object tracker of SAM 2 and the SAM 3 tracker. |
| [florence2/](./florence2/README.md) | Florence-2 detection + region segmentation, referring expression segmentation and box segmentation. |
