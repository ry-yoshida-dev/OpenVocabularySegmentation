import pytest
import torch

from open_vocabulary_segmentation import Device, SegmentationThresholds, SegmenterBackend, SegmenterSettings
from open_vocabulary_segmentation.runtime import TorchRuntime


def build_settings(device: Device, is_half_precision_enabled: bool = False) -> SegmenterSettings:
    return SegmenterSettings(
        backend=SegmenterBackend.SAM3,
        weights_path="stub",
        thresholds=SegmentationThresholds(confidence_threshold=0.5, nms_iou_threshold=0.5),
        device=device,
        is_half_precision_enabled=is_half_precision_enabled,
    )


def test_cpu_runtime_prepares_models_and_inputs() -> None:
    runtime: TorchRuntime = TorchRuntime(build_settings(Device.CPU))
    assert runtime.device == torch.device("cpu")
    assert runtime.dtype == torch.float32
    model: torch.nn.Linear = torch.nn.Linear(2, 2).double()
    runtime.prepare_model(model)
    assert not model.training
    assert model.weight.dtype == torch.float32
    assert runtime.to_model_input(torch.zeros(1, 2, dtype=torch.float64)).dtype == torch.float32


def test_auto_device_resolves_to_an_available_device() -> None:
    runtime: TorchRuntime = TorchRuntime(build_settings(Device.AUTO))
    assert runtime.device.type in {Device.CPU, Device.CUDA, Device.MPS}


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_unavailable_cuda_raises() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        TorchRuntime(build_settings(Device.CUDA))


def test_half_precision_on_cpu_raises() -> None:
    with pytest.raises(ValueError, match="half precision"):
        TorchRuntime(build_settings(Device.CPU, is_half_precision_enabled=True))
