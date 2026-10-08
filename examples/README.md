# examples

## Overview

Runnable scripts using the package.

## Components

| Component | Description |
| --------- | ----------- |
| [segment_directory.py](./segment_directory.py) | Segments classes in every image of a directory with any backend and weights, prints the instances and saves mask overlays; `name:query,query` gives a class several text queries. |

## Examples

```bash
python examples/segment_directory.py images/ overlays/ --backend florence2 --weights florence-community/Florence-2-base \
    --classes person bicycle
python examples/segment_directory.py images/ overlays/ --backend sam --weights facebook/sam-vit-base \
    --box-detector-backend owl_vit --box-detector-weights google/owlv2-base-patch16-ensemble \
    --box-confidence-threshold 0.2 --classes "car:car,suv,taxi" person
```
