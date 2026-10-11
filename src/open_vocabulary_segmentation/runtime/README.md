# runtime

## Overview

How a model is executed. The segmenter, generator and tracker bases are independent of any inference framework;
each implementation holds the runtime its model needs. Every current backend runs on PyTorch through `TorchRuntime`.
The box detector of a box-prompted backend runs on its own runtime from `open_vocabulary_detector`.

## Components

| Component | Description |
| --------- | ----------- |
| [pytorch.py](./pytorch.py) | `TorchRuntime`: resolves `Device` to a `torch.device`, maps `Precision` to a dtype, and prepares models and inputs. |

## Examples

```python
runtime = TorchRuntime(Device.AUTO, Precision.BFLOAT16)
runtime.prepare_model(model)
pixel_values = runtime.to_model_input(pixel_values)
labels = runtime.to_device(labels)
```
