# tests

## Overview

Unit tests that run without downloading model weights, plus smoke tests of every backend on real checkpoints
(marked `slow`, skipped unless `pytest --run-slow` is given).

## Components

| Component | Description |
| --------- | ----------- |
| [conftest.py](./conftest.py) | Shared fixture (a prompt mixing text and visual queries) and the `--run-slow` option. |
| [test_configuration.py](./test_configuration.py) | YAML presets matching the `SegmenterSettings` and nested `DetectorSettings` schemas, backend capabilities and settings validation. |
| [test_segmenter.py](./test_segmenter.py) | Shared mini-batching, post-processing (threshold, mask NMS, sorting), merging of queries per class, `ImageContainer` inputs, query kind and backend checks. |
| [test_segmentation_result.py](./test_segmentation_result.py) | `SegmentationResult` construction, boxes from masks, selection by class, query, area and rank, mask NMS, class and union masks, label maps, counts, and `MaskOverlap`. |
| [test_segmentation.py](./test_segmentation.py) | `Segmentation` crop slices, crops keeping the container type, cutouts in RGB, size checks and polygons. |
| [test_overlay_renderer.py](./test_overlay_renderer.py) | `MaskOverlayRenderer` tinting, blending, outlines and validation. |
| [test_box_prompted.py](./test_box_prompted.py) | Detector boxes turned into masks with the detector's confidences and queries, skipped images, box chunking and checkpoint checks. |
| [test_florence2_polygons.py](./test_florence2_polygons.py) | Florence-2 polygon parsing, rasterization and location tokens of region prompts. |
| [test_sam3_prompt_encoding.py](./test_sam3_prompt_encoding.py) | SAM 3 text features repeated per image, reuse per prompt and validation. |
| [test_torch_runtime.py](./test_torch_runtime.py) | `TorchRuntime` device resolution, dtype and model preparation. |
| [test_package_import.py](./test_package_import.py) | Importing the package loads no backend (`transformers` models). |
| [test_real_models.py](./test_real_models.py) | `slow`: each backend loads its real checkpoint and segments a drawn circle; SAM 3 is skipped until its gated weights are cached. |
