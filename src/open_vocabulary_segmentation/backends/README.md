# backends

## Overview

Concrete `OpenVocabularySegmenter` implementations. Every segmenter takes the common `SegmenterSettings`, declares
the `SegmenterBackend` it serves, checks the checkpoint's `model_type` before loading (`CheckpointConfig`), runs its
model through a `TorchRuntime`, and returns raw instances to the shared post-processing of `OpenVocabularySegmenter`,
so thresholding, mask NMS and sorting behave identically across backends.

SAM and SAM 2 share `BoxPromptedSegmenter`, which masks the boxes of an `open_vocabulary_detector` model. SAM 3 and
Florence-2 find instances of text phrases themselves.

Nothing is imported eagerly: `SegmenterSettings.build()` imports only the sub-package of the requested backend, so
importing `open_vocabulary_segmentation` loads no `transformers` model.

## Components

| Component | Description |
| --------- | ----------- |
| [checkpoint_config.py](./checkpoint_config.py) | `CheckpointConfig`: reads and checks the `model_type` of a `transformers` checkpoint without loading its weights. |
| [box_prompted/](./box_prompted/README.md) | `BoxPromptedSegmenter`: shared base masking detector boxes. |
| [sam/](./sam/README.md) | SAM; masks detector boxes. |
| [sam2/](./sam2/README.md) | SAM 2 and SAM 2.1; masks detector boxes. |
| [sam3/](./sam3/README.md) | SAM 3; text queries. |
| [florence2/](./florence2/README.md) | Florence-2 detection followed by region segmentation; text queries, no scores. |
