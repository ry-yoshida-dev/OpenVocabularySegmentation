# config

## Overview

Ready-made YAML presets of every supported model, named by model variant and size. Each preset's section maps
one-to-one onto the fields of a settings dataclass (enums written by value, nested dataclasses as nested sections,
`null` for absent values), so it can be built with any dataclass builder such as
`DictConfigHandler.build_dataclass(SegmenterSettings, key="segmenter")`.

| Folder | Section | Settings | Presets |
| ------ | ------- | -------- | ------- |
| [sam/](./sam) | `segmenter` | `SegmenterSettings` | SAM `vit_base`, `vit_large`, `vit_huge` + Grounding DINO tiny |
| [sam2/](./sam2) | `segmenter` | `SegmenterSettings` | SAM 2.1 `hiera_tiny`, `hiera_small`, `hiera_base_plus`, `hiera_large` + Grounding DINO tiny |
| [sam3/](./sam3) | `segmenter` | `SegmenterSettings` | SAM 3 `original` (gated weights); SAM 3-LiteText `litetext_s0`, `litetext_s1`, `litetext_l` (light MobileCLIP-style text encoders) |
| [edgetam/](./edgetam) | `segmenter` | `SegmenterSettings` | EdgeTAM `original` + Grounding DINO tiny |
| [florence2/](./florence2) | `segmenter` | `SegmenterSettings` | `base`, `base_ft`, `large`, `large_ft` |
| [florence2_referring/](./florence2_referring) | `segmenter` | `SegmenterSettings` | `base`, `base_ft`, `large`, `large_ft` |
| [promptable/](./promptable) | `promptable_segmenter` | `PromptableSegmenterSettings` | `sam/`, `sam2/`, `sam3/`, `edgetam/`, `florence2/` variants |
| [tracker/](./tracker) | `object_tracker` | `ObjectTrackerSettings` | `sam2/` SAM 2.1 variants, `sam3/original`, `edgetam/original` |
| [concept_tracker/](./concept_tracker) | `concept_tracker` | `ConceptTrackerSettings` | `sam3/original` |
| [generation/](./generation) | `mask_generator` | `MaskGeneratorSettings` | `sam/`, `sam2/`, `sam3/`, `edgetam/` variants with SAM's (SAM) or SAM 2's (SAM 2, SAM 3, EdgeTAM) defaults |

Box-prompted presets (`sam`, `sam2`, `edgetam`) carry a `box_detector` section in the format of
`open_vocabulary_detector.DetectorSettings`, using Grounding DINO tiny; replace it with any detector preset of that
package. Every preset runs in `float32`; on a GPU, `bfloat16` is the precision SAM 2 and SAM 3 were trained in.
A preset is a starting point to copy into an application config.

## Examples

```yaml
promptable_segmenter:
  backend: sam2
  weights_path: facebook/sam2.1-hiera-tiny
  batch_size: 4
  decode_batch_size: 64
  device: auto
  precision: float32
```

```python
settings = DictConfigHandler(cfg=OmegaConf.load(path)).build_dataclass(ObjectTrackerSettings, key="object_tracker")
```
