from collections.abc import Sequence
from typing import ClassVar, cast

import torch
from open_vocabulary_detector import Device, ImageSize, Prompt, PromptKind
from PIL import Image
from transformers import Sam3Model, Sam3VideoModel, Sam3VideoProcessor
from transformers.models.sam3_video.modeling_sam3_video import Sam3VideoInferenceSession

from ...runtime import TorchRuntime
from ...video.concept_tracking import (
    ConceptTracker,
    ConceptTrackerSettings,
    ConceptTrackingParameters,
    ConceptTrackingSession,
)
from ..checkpoint_config import CheckpointConfig
from .concept_tracking_session import Sam3ConceptTrackingSession
from .prompt_encoder import PromptEncoder


class Sam3ConceptTracker(ConceptTracker):
    """
    SAM 3 video concept tracker backed by ``transformers.Sam3VideoModel``.

    SAM 3 video runs the SAM 3 detector on every frame and the SAM 3 tracker on every track, matches detections with
    tracks, starts tracks for new instances and removes duplicate and spurious tracks during a hot start. Text and
    visual queries are encoded by ``PromptEncoder`` for the detector, so reference images define concepts as on
    still images.
    ``ConceptTrackingParameters`` override the thresholds of this fusion on the loaded model; a hot start shorter
    than its frame counts is rejected, since it would silently disable the removal. Mask post-processing
    (hole filling, speck removal, NMS) needs the optional ``kernels`` package; without it, ``transformers`` skips it.
    """

    MODEL_TYPES: ClassVar[frozenset[str]] = frozenset({"sam3_video"})

    def __init__(self, settings: ConceptTrackerSettings) -> None:
        """
        Check the checkpoint, then load the processor and model and apply the parameter overrides.

        Parameters
        ----------
        settings : ConceptTrackerSettings
            Checkpoint, thresholds, device and precision.

        Raises
        ------
        ValueError
            If the checkpoint is not a ``sam3_video`` checkpoint.
        TypeError
            If the checkpoint does not hold a SAM 3 detector.
        """
        super().__init__(settings)
        CheckpointConfig.require_model_type(settings.weights_path, self.MODEL_TYPES)
        self._runtime: TorchRuntime = TorchRuntime(settings.device, settings.precision)
        self._processor: Sam3VideoProcessor = Sam3VideoProcessor.from_pretrained(settings.weights_path)
        self._model: Sam3VideoModel = Sam3VideoModel.from_pretrained(settings.weights_path)
        self._runtime.prepare_model(self._model)
        detector: object = cast(object, self._model.detector_model)
        if not isinstance(detector, Sam3Model):
            raise TypeError(f"unexpected SAM 3 video detector: {type(detector)}")
        self._prompt_encoder: PromptEncoder = PromptEncoder(detector, self._processor, self._runtime)
        self._apply(settings.parameters)

    @property
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        """
        Query kinds the tracker accepts.

        Returns
        -------
        frozenset[PromptKind]
            ``TEXT`` and ``VISUAL``.
        """
        return frozenset({PromptKind.TEXT, PromptKind.VISUAL})

    def _apply(self, parameters: ConceptTrackingParameters) -> None:
        model: Sam3VideoModel = self._model
        if parameters.detection_threshold is not None:
            model.score_threshold_detection = parameters.detection_threshold
        if parameters.new_track_threshold is not None:
            model.new_det_thresh = parameters.new_track_threshold
        if parameters.detection_nms_iou_threshold is not None:
            model.det_nms_thresh = parameters.detection_nms_iou_threshold
        if parameters.association_iou_threshold is not None:
            model.assoc_iou_thresh = parameters.association_iou_threshold
        if parameters.track_association_iou_threshold is not None:
            model.trk_assoc_iou_thresh = parameters.track_association_iou_threshold
        if parameters.hotstart_delay is not None:
            model.hotstart_delay = parameters.hotstart_delay
        if parameters.hotstart_unmatched_frame_count is not None:
            model.hotstart_unmatch_thresh = parameters.hotstart_unmatched_frame_count
        if parameters.hotstart_duplicate_frame_count is not None:
            model.hotstart_dup_thresh = parameters.hotstart_duplicate_frame_count
        if parameters.recondition_interval is not None:
            model.recondition_every_nth_frame = parameters.recondition_interval
        if parameters.recondition_confidence_threshold is not None:
            model.high_conf_thresh = parameters.recondition_confidence_threshold
        if parameters.recondition_iou_threshold is not None:
            model.high_iou_thresh = parameters.recondition_iou_threshold
        if parameters.occlusion_suppression_iou_threshold is not None:
            model.suppress_overlapping_based_on_recent_occlusion_threshold = (
                parameters.occlusion_suppression_iou_threshold
            )
        if parameters.max_object_count is not None:
            model.max_num_objects = parameters.max_object_count
        if parameters.fill_hole_area is not None:
            model.fill_hole_area = parameters.fill_hole_area
        hotstart_delay: int = int(model.hotstart_delay)
        hotstart_frame_count: int = max(int(model.hotstart_unmatch_thresh), int(model.hotstart_dup_thresh))
        if hotstart_delay > 0 and hotstart_frame_count > hotstart_delay:
            raise ValueError(
                f"hotstart_delay ({hotstart_delay}) is shorter than the hot start frame counts "
                + f"({hotstart_frame_count}), so no track would ever be removed; lower "
                + "hotstart_unmatched_frame_count and hotstart_duplicate_frame_count too."
            )

    def _start_offline(self, frames: Sequence[Image.Image], prompt: Prompt) -> ConceptTrackingSession:
        frame_device: torch.device = self._device_of(self.settings.is_frame_storage_on_device)
        inference_session: Sam3VideoInferenceSession = self._processor.init_video_session(
            video=list(frames),
            inference_device=self._runtime.device,
            inference_state_device=self._device_of(self.settings.is_tracking_state_on_device),
            processing_device=frame_device,
            video_storage_device=frame_device,
            dtype=self._runtime.dtype,
        )
        return self._session_of(inference_session, prompt, ImageSize.from_image(frames[0]), len(frames))

    def _start_streaming(self, prompt: Prompt) -> ConceptTrackingSession:
        inference_session: Sam3VideoInferenceSession = self._processor.init_video_session(
            inference_device=self._runtime.device, dtype=self._runtime.dtype
        )
        return self._session_of(inference_session, prompt, None, 0)

    def _session_of(
        self,
        inference_session: Sam3VideoInferenceSession,
        prompt: Prompt,
        image_size: ImageSize | None,
        frame_count: int,
    ) -> ConceptTrackingSession:
        return Sam3ConceptTrackingSession(
            model=self._model,
            processor=self._processor,
            inference_session=inference_session,
            runtime=self._runtime,
            prompt_encoding=self._prompt_encoder.encode(prompt),
            settings=self.settings,
            image_size=image_size,
            frame_count=frame_count,
        )

    def _device_of(self, is_on_device: bool) -> torch.device:
        return self._runtime.device if is_on_device else torch.device(Device.CPU)
