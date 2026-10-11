import torch
from open_vocabulary_detector import Device

from ..options import Precision


class TorchRuntime:
    """
    Runs PyTorch models on a requested device and precision.

    Segmenters and trackers backed by PyTorch hold one; a model backed by another runtime (ONNX Runtime, TensorRT)
    would hold its own runtime instead.
    """

    def __init__(self, device: Device, precision: Precision) -> None:
        """
        Parameters
        ----------
        device : Device
            Device to run on; ``AUTO`` picks CUDA, then MPS, then the CPU.
        precision : Precision
            Precision of the model weights and floating-point inputs.

        Raises
        ------
        RuntimeError
            If an explicitly requested GPU backend is unavailable.
        ValueError
            If the precision is not supported on the CPU and the device resolves to the CPU.
        """
        self._device: torch.device = self._resolve_device(device)
        if self._device.type == Device.CPU and not precision.is_cpu_supported:
            raise ValueError(f"{precision} precision is not supported on the CPU.")
        self._precision: Precision = precision
        self._dtype: torch.dtype = self._resolve_dtype(precision)

    @property
    def device(self) -> torch.device:
        """
        Device the model runs on.

        Returns
        -------
        torch.device
            Resolved device.
        """
        return self._device

    @property
    def precision(self) -> Precision:
        """
        Precision of the model weights and floating-point inputs.

        Returns
        -------
        Precision
            Requested precision.
        """
        return self._precision

    @property
    def dtype(self) -> torch.dtype:
        """
        Floating-point type of the model weights and floating-point inputs.

        Returns
        -------
        torch.dtype
            dtype of ``precision``.
        """
        return self._dtype

    def prepare_model(self, model: torch.nn.Module) -> None:
        """
        Put a model in evaluation mode on the runtime device and dtype.

        Parameters
        ----------
        model : torch.nn.Module
            Model to prepare in place.
        """
        model.eval()
        model.to(device=self._device, dtype=self._dtype)

    def to_model_input(self, values: torch.Tensor) -> torch.Tensor:
        """
        Move floating-point model inputs (pixels, coordinates, mask logits) to the runtime device and dtype.

        Parameters
        ----------
        values : torch.Tensor
            Preprocessed floating-point input.

        Returns
        -------
        torch.Tensor
            Input ready for the model.
        """
        return values.to(device=self._device, dtype=self._dtype)

    def to_device(self, values: torch.Tensor) -> torch.Tensor:
        """
        Move non-floating-point model inputs (token ids, labels, attention masks) to the runtime device.

        Parameters
        ----------
        values : torch.Tensor
            Preprocessed input whose dtype must be kept.

        Returns
        -------
        torch.Tensor
            Input on the runtime device.
        """
        return values.to(device=self._device)

    @staticmethod
    def _resolve_dtype(precision: Precision) -> torch.dtype:
        match precision:
            case Precision.FLOAT32:
                return torch.float32
            case Precision.FLOAT16:
                return torch.float16
            case Precision.BFLOAT16:
                return torch.bfloat16

    @staticmethod
    def _resolve_device(device: Device) -> torch.device:
        match device:
            case Device.AUTO:
                if torch.cuda.is_available():
                    return torch.device(Device.CUDA)
                if torch.backends.mps.is_available():
                    return torch.device(Device.MPS)
                return torch.device(Device.CPU)
            case Device.CPU:
                return torch.device(Device.CPU)
            case Device.CUDA:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA was requested but is not available.")
                return torch.device(Device.CUDA)
            case Device.MPS:
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS was requested but is not available.")
                return torch.device(Device.MPS)
