from collections.abc import Iterator, Mapping
from typing import ClassVar, cast

import numpy as np
import torch
from image_container import BinaryImage
from open_vocabulary_detector import ImageSize
from PIL import Image
from torch.nn import functional
from transformers import (
    BatchEncoding,
    EdgeTamVideoModel,
    Sam2VideoInferenceSession,
    Sam2VideoModel,
    Sam2VideoProcessor,
    Sam3TrackerVideoInferenceSession,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
)
from transformers.models.edgetam_video.modeling_edgetam_video import EdgeTamVideoInferenceSession

from ...array_types import BoolArray, FloatArray, IntArray
from ...object_prompt import ObjectPrompt
from ...runtime import TorchRuntime
from ...video.object_tracking import TrackedFrame, TrackingSession
from .history_pruner import HistoryPruner
from .model_types import MemoryInferenceSession, MemoryTrackerModel, MemoryTrackerOutput, MemoryTrackerProcessor


class MemoryTrackingSession(TrackingSession):
    """
    Tracking session of the SAM 2 video predictor, the SAM 3 tracker and EdgeTAM, backed by a ``transformers``
    inference session.

    EdgeTAM is driven by the SAM 2 video processor, whose ``Sam2VideoInferenceSession`` is what EdgeTAM reads although
    ``transformers`` annotates it with its identical ``EdgeTamVideoInferenceSession`` copy.

    A stream frees the history the memory never reads again after every frame (``HistoryPruner``). The prompts of
    one frame are all added before a single forward pass and marked as new together, since a ``transformers`` input
    only marks its own objects as new; every prompted object then conditions on the frame, whatever the prompt
    order. ``transformers`` stores a prompt on a frame the object was already tracked on as a tracking output, which
    the next propagation recomputes without the prompt; such outputs are moved to the conditioning outputs, like
    ``add_all_frames_to_correct_as_cond`` of the original SAM 2, so corrections persist. ``transformers`` skips a
    backward pass from frame 0, so such a pass tracks frame 0 alone. Mask logits are resized to
    the frame and binarized at ``MASK_THRESHOLD`` like the ``transformers`` post-processing.
    """

    POSITIVE_LABEL: ClassVar[int] = 1
    NEGATIVE_LABEL: ClassVar[int] = 0
    MASK_THRESHOLD: ClassVar[float] = 0.0

    def __init__(
        self,
        model: MemoryTrackerModel,
        processor: MemoryTrackerProcessor,
        inference_session: MemoryInferenceSession,
        runtime: TorchRuntime,
        image_size: ImageSize | None,
        frame_count: int,
        conditioning_frame_limit: int,
    ) -> None:
        """
        Parameters
        ----------
        model : MemoryTrackerModel
            Loaded video model in evaluation mode.
        processor : MemoryTrackerProcessor
            Processor of the model.
        inference_session : MemoryInferenceSession
            ``transformers`` session, holding a whole video or empty for a stream.
        runtime : TorchRuntime
            Device and dtype of the model.
        image_size : ImageSize | None
            Size of the frames; ``None`` for a stream.
        frame_count : int
            Number of frames of a whole video; ``0`` for a stream.
        conditioning_frame_limit : int
            Maximum number of prompted frames per object a stream keeps in its memory.
        """
        super().__init__(image_size=image_size, frame_count=frame_count, is_streaming=image_size is None)
        self._model: MemoryTrackerModel = model
        self._processor: MemoryTrackerProcessor = processor
        self._inference_session: MemoryInferenceSession = inference_session
        self._runtime: TorchRuntime = runtime
        self._pruner: HistoryPruner = HistoryPruner(
            history_length=max(int(model.config.num_maskmem), int(model.config.max_object_pointers_in_encoder)),
            conditioning_frame_limit=conditioning_frame_limit,
        )

    def _prompt(self, frame_index: int, prompts: Mapping[int, ObjectPrompt]) -> TrackedFrame:
        image_size: ImageSize = self._require_image_size()
        self._add_all_inputs(frame_index, prompts, image_size)
        output: MemoryTrackerOutput = self._run(frame_index, None)
        self._keep_as_conditioning(frame_index, prompts)
        return self._tracked_frame_of(output, image_size)

    def _propagate(
        self, is_reverse: bool, start_frame_index: int | None, max_frame_count: int | None
    ) -> Iterator[TrackedFrame]:
        image_size: ImageSize = self._require_image_size()
        if is_reverse and self._resolved_start_frame_index(start_frame_index) == 0:
            yield self._tracked_frame_of(self._run(0, None, is_reverse=True), image_size)
            return
        outputs: Iterator[MemoryTrackerOutput]
        match (self._model, self._inference_session):
            case (Sam3TrackerVideoModel() as model, Sam3TrackerVideoInferenceSession() as inference_session):
                outputs = model.propagate_in_video_iterator(
                    inference_session,
                    start_frame_idx=start_frame_index,
                    max_frame_num_to_track=max_frame_count,
                    reverse=is_reverse,
                )
            case (Sam2VideoModel() as model, Sam2VideoInferenceSession() as inference_session):
                outputs = model.propagate_in_video_iterator(
                    inference_session,
                    start_frame_idx=start_frame_index,
                    max_frame_num_to_track=max_frame_count,
                    reverse=is_reverse,
                )
            case (EdgeTamVideoModel() as model, Sam2VideoInferenceSession() as inference_session):
                outputs = model.propagate_in_video_iterator(
                    cast(EdgeTamVideoInferenceSession, inference_session),
                    start_frame_idx=start_frame_index,
                    max_frame_num_to_track=max_frame_count,
                    reverse=is_reverse,
                )
            case _:
                raise TypeError(
                    f"{type(self._model).__name__} cannot drive a {type(self._inference_session).__name__}."
                )
        for output in outputs:
            yield self._tracked_frame_of(output, image_size)

    def _track_next(self, frame: Image.Image, frame_index: int, prompts: Mapping[int, ObjectPrompt]) -> TrackedFrame:
        image_size: ImageSize = self._require_image_size()
        inputs: BatchEncoding = self._processor(images=frame, return_tensors="pt")
        pixel_values: torch.Tensor = self._runtime.to_model_input(inputs["pixel_values"][0])
        if not self._object_ids:
            self._inference_session.add_new_frame(pixel_values, frame_index)
            self._prune(frame_index)
            return TrackedFrame.empty(frame_index, image_size)
        self._add_all_inputs(frame_index, prompts, image_size)
        output: MemoryTrackerOutput = self._run(frame_index, pixel_values)
        self._keep_as_conditioning(frame_index, prompts)
        self._prune(frame_index)
        return self._tracked_frame_of(output, image_size)

    def _reset(self) -> None:
        match self._inference_session:
            case Sam3TrackerVideoInferenceSession() as inference_session:
                inference_session.reset_tracking_data()
            case Sam2VideoInferenceSession() as inference_session:
                inference_session.reset_tracking_data()

    def _add_all_inputs(self, frame_index: int, prompts: Mapping[int, ObjectPrompt], image_size: ImageSize) -> None:
        for object_id, prompt in prompts.items():
            self._add_inputs(frame_index, object_id, prompt, image_size)
        if prompts:
            self._inference_session.obj_with_new_inputs = list(prompts)

    def _add_inputs(self, frame_index: int, object_id: int, prompt: ObjectPrompt, image_size: ImageSize) -> None:
        original_size: tuple[int, int] = (image_size.height, image_size.width)
        if isinstance(prompt.mask, BinaryImage):
            self._add_to_session(
                frame_index,
                object_id,
                original_size,
                input_masks=np.asarray(prompt.mask.value, dtype=np.bool_),
            )
            return
        point_xy: FloatArray = prompt.point_xy
        labels: IntArray = np.where(prompt.is_point_positive, self.POSITIVE_LABEL, self.NEGATIVE_LABEL).astype(np.int64)
        xyxy: FloatArray | None = prompt.xyxy
        self._add_to_session(
            frame_index,
            object_id,
            original_size,
            input_points=[[[[float(x), float(y)] for x, y in point_xy]]] if len(point_xy) else None,
            input_labels=[[[int(label) for label in labels]]] if len(point_xy) else None,
            input_boxes=None if xyxy is None else [[[float(value) for value in xyxy]]],
        )

    def _add_to_session(
        self,
        frame_index: int,
        object_id: int,
        original_size: tuple[int, int],
        input_points: list[list[list[list[float]]]] | None = None,
        input_labels: list[list[list[int]]] | None = None,
        input_boxes: list[list[list[float]]] | None = None,
        input_masks: BoolArray | None = None,
    ) -> None:
        match (self._processor, self._inference_session):
            case (Sam3TrackerVideoProcessor() as processor, Sam3TrackerVideoInferenceSession() as inference_session):
                processor.add_inputs_to_inference_session(
                    inference_session,
                    frame_idx=frame_index,
                    obj_ids=[object_id],
                    input_points=input_points,
                    input_labels=input_labels,
                    input_boxes=input_boxes,
                    input_masks=input_masks,
                    original_size=original_size,
                )
            case (Sam2VideoProcessor() as processor, Sam2VideoInferenceSession() as inference_session):
                processor.add_inputs_to_inference_session(
                    inference_session,
                    frame_idx=frame_index,
                    obj_ids=[object_id],
                    input_points=input_points,
                    input_labels=input_labels,
                    input_boxes=input_boxes,
                    input_masks=input_masks,
                    original_size=original_size,
                )
            case _:
                raise TypeError(
                    f"{type(self._processor).__name__} cannot drive a {type(self._inference_session).__name__}."
                )

    def _run(
        self, frame_index: int, pixel_values: torch.Tensor | None, is_reverse: bool = False
    ) -> MemoryTrackerOutput:
        with torch.inference_mode():
            output: MemoryTrackerOutput = self._model(
                inference_session=self._inference_session, frame_idx=frame_index, frame=pixel_values, reverse=is_reverse
            )
        return output

    def _resolved_start_frame_index(self, start_frame_index: int | None) -> int | None:
        if start_frame_index is not None:
            return start_frame_index
        output_dicts: dict[int, dict[str, dict[int, object]]] = self._inference_session.output_dict_per_obj
        conditioning_frame_indices: list[int] = [
            frame_index
            for outputs in output_dicts.values()
            for frame_index in outputs[HistoryPruner.CONDITIONING_OUTPUTS_KEY]
        ]
        return min(conditioning_frame_indices, default=None)

    def _tracked_frame_of(self, output: MemoryTrackerOutput, image_size: ImageSize) -> TrackedFrame:
        object_ids: list[int] = list(output.object_ids or [])
        frame_index: int = int(output.frame_idx if output.frame_idx is not None else 0)
        if not object_ids or output.pred_masks is None:
            return TrackedFrame.empty(frame_index, image_size)
        upscaled: torch.Tensor = functional.interpolate(
            output.pred_masks.float(), size=(image_size.height, image_size.width), mode="bilinear", align_corners=False
        )
        binary_masks: BoolArray = (upscaled[:, 0] > self.MASK_THRESHOLD).cpu().numpy()
        presence_scores: FloatArray = (
            np.ones(len(object_ids), dtype=np.float64)
            if output.object_score_logits is None
            else output.object_score_logits.float().sigmoid().reshape(-1).cpu().numpy().astype(np.float64)
        )
        return TrackedFrame(
            frame_index=frame_index,
            object_ids=np.array(object_ids, dtype=np.int64),
            masks=binary_masks,
            presence_scores=presence_scores,
            image_size=image_size,
        )

    def _keep_as_conditioning(self, frame_index: int, prompts: Mapping[int, ObjectPrompt]) -> None:
        output_dicts: dict[int, dict[str, dict[int, object]]] = self._inference_session.output_dict_per_obj
        for object_id in prompts:
            outputs: dict[str, dict[int, object]] = output_dicts[self._inference_session.obj_id_to_idx(object_id)]
            tracking_outputs: dict[int, object] = outputs[HistoryPruner.NON_CONDITIONING_OUTPUTS_KEY]
            if frame_index in tracking_outputs:
                outputs[HistoryPruner.CONDITIONING_OUTPUTS_KEY][frame_index] = tracking_outputs.pop(frame_index)

    def _prune(self, frame_index: int) -> None:
        output_dicts: dict[int, dict[str, dict[int, object]]] = self._inference_session.output_dict_per_obj
        tracked_frames_per_object: dict[int, dict[int, object]] = self._inference_session.frames_tracked_per_obj
        point_inputs: dict[int, dict[int, object]] = self._inference_session.point_inputs_per_obj
        mask_inputs: dict[int, dict[int, object]] = self._inference_session.mask_inputs_per_obj
        self._pruner.prune(
            frame_index,
            self._inference_session.processed_frames,
            output_dicts,
            tracked_frames_per_object,
            (point_inputs, mask_inputs),
        )
