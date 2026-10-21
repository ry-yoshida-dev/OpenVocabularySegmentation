# tests

## Overview

Unit tests that run without downloading model weights, plus smoke tests of every backend on real checkpoints
(marked `slow`, skipped unless `pytest --run-slow` is given).

## Components

| Component | Description |
| --------- | ----------- |
| [conftest.py](./conftest.py) | Shared fixture (a prompt mixing text and visual queries) and the `--run-slow` option. |
| [stub_promptable.py](./stub_promptable.py) | `StubPromptableSegmenter`: a SAM-family mask decoder painting boxes and points into its logits, and checkpoint patching. |
| [test_promptable_segmenter.py](./test_promptable_segmenter.py) | Prompt mapping into the model frame, head selection per `MaskOutput`, grouping and passes, encoding reuse, refinement and validation. |
| [test_object_prompt.py](./test_object_prompt.py) | `ObjectPrompt`, `MaskLogits`, `ModelFrame`, `PromptSignature`, `PromptTensors`, `DecodedMasks` and `MaskOutput`. |
| [test_object_segmentation_result.py](./test_object_segmentation_result.py) | `ObjectSegmentationResult` items, filters, best candidates, NMS and maps; mask bounds and suppression; `SemanticSegmentationResult`. |
| [test_exemplars.py](./test_exemplars.py) | `BoxExemplars`, `ImageExemplars`, SAM 3 exemplar tensors and prediction accumulation. |
| [test_mask_generation.py](./test_mask_generation.py) | Crop layout, point grids, cut detection, region cleanup, generation options, the generator and its result. |
| [test_video_tracking.py](./test_video_tracking.py) | `TrackedFrame`, `ConceptFrame`, offline and streaming session validation, propagation passes, resets and stream history pruning. |
| [test_rgb_image.py](./test_rgb_image.py) | `RgbImage` conversion of RGB, channel-ordered containers and high bit depth grayscale images. |
| [test_configuration.py](./test_configuration.py) | Every YAML preset matching its settings schema recursively, preset coverage of every backend, backend capabilities and settings validation. |
| [test_segmenter.py](./test_segmenter.py) | Shared mini-batching, post-processing (threshold, mask NMS, sorting), merging of queries per class, `ImageContainer` inputs, exemplar validation, query kind and backend checks. |
| [test_segmentation_result.py](./test_segmentation_result.py) | `SegmentationResult` construction, boxes from masks, selection by class, query, area and rank, mask NMS, class and union masks, label maps, counts, and `MaskOverlap`. |
| [test_segmentation.py](./test_segmentation.py) | `Segmentation` crop slices, crops keeping the container type, cutouts in RGB, size checks and polygons. |
| [test_overlay_renderer.py](./test_overlay_renderer.py) | `MaskOverlayRenderer` tinting, blending, outlines, prompts and every result type. |
| [test_box_prompted.py](./test_box_prompted.py) | Detector boxes masked by a promptable segmenter with the detector's confidences and queries, skipped images, shared passes, clipping and checkpoint checks. |
| [test_florence2_polygons.py](./test_florence2_polygons.py) | Florence-2 polygon parsing, rasterization and location tokens of region prompts. |
| [stub_sam3.py](./stub_sam3.py) | `StubSam3`: a SAM 3 detector and processor whose features reveal texts and geometry calls. |
| [test_sam3_prompt_encoder.py](./test_sam3_prompt_encoder.py) | SAM 3 visual queries as the visual text plus reference geometry, padding and geometry reuse. |
| [test_sam3_prompt_encoding.py](./test_sam3_prompt_encoding.py) | SAM 3 text features repeated per image, reuse per prompt and validation. |
| [test_torch_runtime.py](./test_torch_runtime.py) | `TorchRuntime` device resolution, precisions and model preparation. |
| [test_package_import.py](./test_package_import.py) | Importing the package loads no backend (`transformers` models). |
| [test_real_models.py](./test_real_models.py) | `slow`: every interface on real checkpoints: open-vocabulary backends, promptable segmentation with refinement, Florence-2 boxes, mask generation, object tracking with corrections and backward passes, concept tracking with track removal, SAM 3 exemplars and semantics; masks are checked by IoU and blank images must stay empty; SAM 3 is skipped until its gated weights are cached. |
