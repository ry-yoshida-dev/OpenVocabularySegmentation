# memory_tracker

## Overview

`ObjectTracker` backed by the `transformers` video predictors with a memory of past frames: SAM 2 / SAM 2.1
(`Sam2VideoModel`, `sam2_video` checkpoints), the SAM 3 tracker (`Sam3TrackerVideoModel`, loaded from the
`sam3_video` checkpoint) and EdgeTAM (`EdgeTamVideoModel`, `edgetam_video`, driven by the SAM 2 video processor and
session). They share one implementation; calls that pair a processor or model with its own inference session
dispatch on the pair.

The prompts of one frame are added together and marked as new before a single forward pass, so every prompted object
conditions on the frame whatever the prompt order. A prompt on a frame the object was already tracked on is stored
by `transformers` as a tracking output that the next propagation would recompute without it; it is moved to the
conditioning outputs (SAM 2's `add_all_frames_to_correct_as_cond`), so corrections persist. Offline sessions
preprocess frames and keep outputs on the devices chosen in the settings (the CPU by default).

In a stream, `HistoryPruner` deletes non-conditioning memories older than
`max(num_maskmem, max_object_pointers_in_encoder)` frames, the only history the model reads, keeps the first and
latest conditioning frames up to `stream_conditioning_frame_limit` per object, deletes point and mask inputs of past
frames, and replaces processed frame pixels by empty tensors (the frame count bounds the object pointers read), so
memory stays bounded. SAM 3 video concept tracking uses the same pruner. Masks are resized to the frame and binarized at logit 0; presence is the sigmoid of the object score.

## Components

| Component | Description |
| --------- | ----------- |
| [tracker.py](./tracker.py) | `MemoryObjectTracker`: loads the model of the backend and opens offline and streaming sessions. |
| [session.py](./session.py) | `MemoryTrackingSession`: prompts, propagation, streaming with pruning, and frame results. |
| [history_pruner.py](./history_pruner.py) | `HistoryPruner`: frees streamed history, surplus conditioning frames and past inputs, keeping the frame count. |
| [model_types.py](./model_types.py) | Type aliases of the SAM 2, SAM 3 tracker and EdgeTAM models, processors, sessions and outputs. |
