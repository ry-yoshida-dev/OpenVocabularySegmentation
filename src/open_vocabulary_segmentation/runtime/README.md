# runtime

## Overview

How a segmenter's model is executed. `OpenVocabularySegmenter` itself is independent of any inference framework;
each segmenter holds the runtime its model needs. Every current backend runs on PyTorch through `TorchRuntime`. The
box detector of a box-prompted backend runs on its own runtime from `open_vocabulary_detector`.

## Components

| Component | Description |
| --------- | ----------- |
| [pytorch.py](./pytorch.py) | `TorchRuntime`: resolves `Device` to a `torch.device`, picks the dtype, and prepares models and floating-point inputs. |

## Examples

```python
runtime = TorchRuntime(settings)
runtime.prepare_model(model)
pixel_values = runtime.to_model_input(pixel_values)
```
