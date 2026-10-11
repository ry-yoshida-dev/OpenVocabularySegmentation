# generation

## Overview

Automatic mask generation ("segment everything"), following SAM's `SamAutomaticMaskGenerator` and working with every
SAM-family promptable segmenter (SAM, SAM 2, SAM 3, EdgeTAM). A grid of single-point prompts covers the image and, with crop
layers, overlapping crops at finer scales; every crop is encoded once and prompted in batches. Masks are kept by
predicted IoU and stability, masks cut by an inner crop edge are dropped, duplicates are removed by box NMS within a
crop and across crops (preferring smaller crops), and small holes and islands can be removed.

## Components

| Component | Description |
| --------- | ----------- |
| [generator.py](./generator.py) | `AutomaticMaskGenerator`: prompts the point grids of every crop and filters, deduplicates and cleans the masks. |
| [options.py](./options.py) | `MaskGenerationOptions`: grid, batch, threshold, NMS, crop and cleanup parameters with SAM's names and defaults. |
| [settings.py](./settings.py) | `MaskGeneratorSettings`: promptable segmenter settings and options; `build()` loads the generator. |
| [crop_box.py](./crop_box.py) | `CropBox`: crop layout of every layer, point grids and detection of masks cut by a crop. |
| [crop_candidates.py](./crop_candidates.py) | `CropCandidates`: masks of one crop collected before NMS. |
| [region_cleaner.py](./region_cleaner.py) | `RegionCleaner`: removes small holes and islands from a mask. |
| [region_kind.py](./region_kind.py) | `RegionKind`: `HOLE` or `ISLAND`. |

## Examples

```python
generator = MaskGeneratorSettings(
    segmenter=PromptableSegmenterSettings(backend=PromptableBackend.SAM2, weights_path="facebook/sam2.1-hiera-large"),
    options=MaskGenerationOptions(points_per_side=32, crop_layer_count=1, minimum_region_area=100),
).build()
regions = generator.generate(image).sort_by_area()
MaskOverlayRenderer().render_generated(source, regions).save("everything.png")
```
