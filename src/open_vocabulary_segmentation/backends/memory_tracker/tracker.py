from collections.abc import Sequence
from typing import ClassVar

import torch
from open_vocabulary_detector import Device, ImageSize
from PIL import Image
from transformers import (
    EdgeTamVideoModel,
    Sam2VideoModel,
    Sam2VideoProcessor,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
)

from ...options import TrackerBackend
from ...runtime import TorchRuntime
from ...video.object_tracking import ObjectTracker, ObjectTrackerSettings, TrackingSession
from ..checkpoint_config import CheckpointConfig
from .model_types import MemoryInferenceSession, MemoryTrackerModel, MemoryTrackerProcessor
from .session import MemoryTrackingSession


class MemoryObjectTracker(ObjectTracker):
    """
    Video object tracker with a memory of past frames: SAM 2 / SAM 2.1 (``Sam2VideoModel``), the SAM 3 tracker
    (``Sam3TrackerVideoModel``) and EdgeTAM (``EdgeTamVideoModel``).

    The SAM 2 and EdgeTAM checkpoints are video checkpoints driven by the SAM 2 video processor; the SAM 3 tracker
    loads from the SAM 3 checkpoint (``sam3_video``), whose detector weights it ignores.
    """

    BACKENDS: ClassVar[frozenset[TrackerBackend]] = frozenset(TrackerBackend)

    def __init__(self, settings: ObjectTrackerSettings) -> None:
        """
        Check the checkpoint, then load the processor and model.

        Parameters
        ----------
        settings : ObjectTrackerSettings
            Backend, checkpoint, device and precision.

        Raises
        ------
        ValueError
            If the checkpoint is not one of the backend's model types.
        """
        super().__init__(settings)
        CheckpointConfig.require_model_type(settings.weights_path, settings.backend.model_types)
        self._runtime: TorchRuntime = TorchRuntime(settings.device, settings.precision)
        self._model: MemoryTrackerModel
        self._processor: MemoryTrackerProcessor
        match settings.backend:
            case TrackerBackend.SAM2:
                self._model = Sam2VideoModel.from_pretrained(settings.weights_path)
                self._processor = Sam2VideoProcessor.from_pretrained(settings.weights_path)
            case TrackerBackend.SAM3:
                self._model = Sam3TrackerVideoModel.from_pretrained(settings.weights_path)
                self._processor = Sam3TrackerVideoProcessor.from_pretrained(settings.weights_path)
            case TrackerBackend.EDGETAM:
                self._model = EdgeTamVideoModel.from_pretrained(settings.weights_path)
                self._processor = Sam2VideoProcessor.from_pretrained(settings.weights_path)
        self._runtime.prepare_model(self._model)

    def _start_offline(self, frames: Sequence[Image.Image]) -> TrackingSession:
        frame_device: torch.device = self._device_of(self.settings.is_frame_storage_on_device)
        inference_session: MemoryInferenceSession = self._processor.init_video_session(
            video=list(frames),
            inference_device=self._runtime.device,
            inference_state_device=self._device_of(self.settings.is_tracking_state_on_device),
            processing_device=frame_device,
            video_storage_device=frame_device,
            dtype=self._runtime.dtype,
        )
        return MemoryTrackingSession(
            model=self._model,
            processor=self._processor,
            inference_session=inference_session,
            runtime=self._runtime,
            image_size=ImageSize.from_image(frames[0]),
            frame_count=len(frames),
            conditioning_frame_limit=self.settings.stream_conditioning_frame_limit,
        )

    def _start_streaming(self) -> TrackingSession:
        inference_session: MemoryInferenceSession = self._processor.init_video_session(
            inference_device=self._runtime.device, dtype=self._runtime.dtype
        )
        return MemoryTrackingSession(
            model=self._model,
            processor=self._processor,
            inference_session=inference_session,
            runtime=self._runtime,
            image_size=None,
            frame_count=0,
            conditioning_frame_limit=self.settings.stream_conditioning_frame_limit,
        )

    def _device_of(self, is_on_device: bool) -> torch.device:
        return self._runtime.device if is_on_device else torch.device(Device.CPU)
