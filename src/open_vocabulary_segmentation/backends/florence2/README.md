# florence2

## Overview

Florence-2 through `transformers.Florence2ForConditionalGeneration`, a generative model writing boxes and polygons
as location tokens. One `Florence2Engine` runs the three segmentation-related tasks, each a beam search
(`BEAM_COUNT` = 3 beams, up to `MAX_NEW_TOKENS` = 1024 tokens):

| Task | Used by | Output |
| ---- | ------- | ------ |
| `<OPEN_VOCABULARY_DETECTION>` then `<REGION_TO_SEGMENTATION>` | `Florence2Segmenter` (`SegmenterBackend.FLORENCE2`) | Every instance of each phrase |
| `<REFERRING_EXPRESSION_SEGMENTATION>` | `Florence2ReferringSegmenter` (`SegmenterBackend.FLORENCE2_REFERRING`) | The one object each phrase refers to |
| `<REGION_TO_SEGMENTATION>` | `Florence2PromptableSegmenter` (`PromptableBackend.FLORENCE2`) | The object in each box |

Images are preprocessed once per mini-batch and shared by all generations; region generations are batched.
Polygons are read at bin centers and filled into full-resolution masks. The model gives no score: every instance has
confidence `1.0`, and a phrase absent from the image usually still gets an instance.

## Components

| Component | Description |
| --------- | ----------- |
| [engine.py](./engine.py) | `Florence2Engine`: model loading, preprocessing, detection, region and referring segmentation. |
| [segmenter.py](./segmenter.py) | `Florence2Segmenter`: detection followed by region segmentation. |
| [referring_segmenter.py](./referring_segmenter.py) | `Florence2ReferringSegmenter`: one referred object per phrase. |
| [promptable_segmenter.py](./promptable_segmenter.py) | `Florence2PromptableSegmenter`: box prompts without scores or logits. |
| [detected_region.py](./detected_region.py) | `DetectedRegion`: box of one query in one image, waiting to be segmented. |
| [location_tokens.py](./location_tokens.py) | `LocationTokens`: writes a box as the location tokens of a region task. |
| [polygon_parser.py](./polygon_parser.py) | `PolygonParser`: reads generated polygons into normalized points. |
| [polygon_rasterizer.py](./polygon_rasterizer.py) | `PolygonRasterizer`: fills the pixels whose centres lie inside polygons, outlining slivers. |
