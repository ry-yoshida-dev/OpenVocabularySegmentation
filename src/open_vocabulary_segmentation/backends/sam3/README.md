# sam3

## Overview

Every capability of SAM 3 in `transformers`, from one checkpoint (`facebook/sam3`, gated on the Hugging Face Hub,
`model_type` `sam3_video`):

- **Concept segmentation** (`Sam3Segmenter`, `Sam3Model`): every instance of a text or visual query. A visual query
  reads the text `"visual"` followed by the geometry tokens of its reference boxes, encoded on each reference image
  by the SAM 3 geometry encoder (`PromptEncoder`). Concepts can be refined by positive and negative box exemplars
  drawn on the image, or defined by boxes alone (the text `"visual"`). The
  semantic head gives a probability map per class (`segment_semantic_images`). Images of a mini-batch are encoded
  once, text queries once per prompt, and the decoders run once per query over the images without exemplars of its
  class and once over those with exemplars (box counts padded with ignored boxes). An instance scores its query match
  times the presence of the concept; masks are binarized at `MASK_THRESHOLD` (0.5). `sam3_lite_text` checkpoints
  load through `Sam3LiteTextModel`.
- **Promptable segmentation** (`Sam3PromptableSegmenter`, `Sam3TrackerModel`): points, boxes and masks, like SAM 2,
  at 1008x1008 with 288x288 mask logits.
- **Video concept tracking** (`Sam3ConceptTracker`, `Sam3VideoModel`): detection and tracking of every instance of
  the text and visual queries with stable track ids and per-frame tracker scores; each query is registered with
  its encoded features under its own prompt key. A backward pass starts on its start frame,
  although `transformers` starts one frame earlier. Streams drop lost tracks and free the per-frame fusion records,
  since SAM 3 removes no track of a stream. Mask post-processing (hole filling, NMS) needs the optional `kernels`
  package.

Video object tracking with the SAM 3 tracker is in [memory_tracker/](../memory_tracker/README.md).
`facebook/sam3.1` ships only the original `.pt` weights, which `transformers` cannot load.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `Sam3Segmenter`: text, visual, exemplar and visual-only concepts; instances and semantic probabilities. |
| [promptable_segmenter.py](./promptable_segmenter.py) | `Sam3PromptableSegmenter`: interactive segmentation with the SAM 3 tracker. |
| [concept_tracker.py](./concept_tracker.py) | `Sam3ConceptTracker`: loads SAM 3 video, applies parameter overrides and opens sessions. |
| [concept_tracking_session.py](./concept_tracking_session.py) | `Sam3ConceptTrackingSession`: propagation, streaming, reset, track removal and frame results. |
| [fusion_state_pruner.py](./fusion_state_pruner.py) | `FusionStatePruner`: frees the per-frame and per-track fusion records of a stream. |
| [prompt_encoder.py](./prompt_encoder.py) | `PromptEncoder`: encodes text and visual queries, caching the geometry of reference images. |
| [prompt_encoding.py](./prompt_encoding.py) | `PromptEncoding`: prompt features of every query of a prompt, reused across image batches. |
| [reference_geometry.py](./reference_geometry.py) | `ReferenceGeometry`: geometry tokens of the boxes of one visual reference. |
| [query_group.py](./query_group.py) | `QueryGroup`: images decoded together for one query, with or without exemplars. |
| [exemplar_tensors.py](./exemplar_tensors.py) | `ExemplarTensors`: exemplar boxes of a group normalized to CXCYWH, labeled and padded. |
| [prediction.py](./prediction.py) | `Sam3Prediction`: instances and semantic probabilities accumulated query by query. |
| [model_outputs.py](./model_outputs.py) | Type aliases of the vision and segmentation outputs of `Sam3Model` and `Sam3LiteTextModel`. |
