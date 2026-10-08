# florence2

## Overview

Florence-2 through `transformers.Florence2ForConditionalGeneration` (`SegmenterBackend.FLORENCE2`). Florence-2
segments one region per generation and its referring-expression segmentation returns a single object per phrase, so
instances are found in two steps with one model:

1. `<OPEN_VOCABULARY_DETECTION>` writes the boxes of every instance of a text query, one generation per query over
   the mini-batch (boxes read by `open_vocabulary_detector`'s `LocationParser`).
2. `<REGION_TO_SEGMENTATION>` writes the polygon outline of each box, generations batched by
   `REGION_CHUNK_SIZE` (8) regions, independently of the image `batch_size`.

Images are preprocessed once per mini-batch and shared by both steps. Polygons are read at bin centers and filled
into full-resolution masks; a box whose polygon encloses no pixel is dropped.

Each generation is a beam search (`BEAM_COUNT` = 3 beams, up to `MAX_NEW_TOKENS` = 1024 tokens), so cost grows with
the number of queries plus the number of instances. The model gives no score: every instance has confidence `1.0`,
and a phrase absent from the image usually still gets an instance.

## Components

| Component | Description |
| --------- | ----------- |
| [segmenter.py](./segmenter.py) | `Florence2Segmenter`. |
| [detected_region.py](./detected_region.py) | `DetectedRegion`: box of one query in one image, waiting to be segmented. |
| [location_tokens.py](./location_tokens.py) | `LocationTokens`: writes a box as the location tokens of a region task. |
| [polygon_parser.py](./polygon_parser.py) | `PolygonParser`: reads generated polygons into normalized points. |
| [polygon_rasterizer.py](./polygon_rasterizer.py) | `PolygonRasterizer`: fills polygons into a binary mask. |
