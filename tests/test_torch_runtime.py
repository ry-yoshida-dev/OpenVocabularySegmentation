import pytest
import torch

from open_vocabulary_segmentation import Device, Precision
from open_vocabulary_segmentation.runtime import TorchRuntime


def test_cpu_runtime_prepares_models_and_inputs() -> None:
    runtime: TorchRuntime = TorchRuntime(Device.CPU, Precision.FLOAT32)
    assert runtime.device == torch.device("cpu")
    assert runtime.dtype == torch.float32
    assert runtime.precision is Precision.FLOAT32
    model: torch.nn.Linear = torch.nn.Linear(2, 2).double()
    runtime.prepare_model(model)
    assert not model.training
    assert model.weight.dtype == torch.float32
    assert runtime.to_model_input(torch.zeros(1, 2, dtype=torch.float64)).dtype == torch.float32
    assert runtime.to_device(torch.zeros(1, 2, dtype=torch.int64)).dtype == torch.int64


def test_bfloat16_runs_on_the_cpu() -> None:
    runtime: TorchRuntime = TorchRuntime(Device.CPU, Precision.BFLOAT16)
    assert runtime.dtype == torch.bfloat16
    assert runtime.to_model_input(torch.zeros(1)).dtype == torch.bfloat16


def test_auto_device_resolves_to_an_available_device() -> None:
    runtime: TorchRuntime = TorchRuntime(Device.AUTO, Precision.FLOAT32)
    assert runtime.device.type in {Device.CPU, Device.CUDA, Device.MPS}


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_unavailable_cuda_raises() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        TorchRuntime(Device.CUDA, Precision.FLOAT32)


def test_half_precision_on_cpu_raises() -> None:
    with pytest.raises(ValueError, match="float16 precision is not supported on the CPU"):
        TorchRuntime(Device.CPU, Precision.FLOAT16)
