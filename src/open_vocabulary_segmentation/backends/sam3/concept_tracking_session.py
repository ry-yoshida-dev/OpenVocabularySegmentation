from collections.abc import Iterator, Mapping
from typing import ClassVar, cast

import numpy as np
import torch
from open_vocabulary_detector import ImageSize
from PIL import Image
from transformers import BatchEncoding, Sam3VideoModel, Sam3VideoProcessor
from transformers.models.sam3_video.modeling_sam3_video import (
    Sam3VideoInferenceSession,
    Sam3VideoSegmentationOutput,
)

from ...array_types import BoolArray, FloatArray, IntArray
from ...result import SegmentationResult
from ...runtime import TorchRuntime
from ...video.concept_tracking import ConceptFrame, ConceptTrackerSettings, ConceptTrackingSession
from ..memory_tracker.history_pruner import HistoryPruner
from .fusion_state_pruner import FusionStatePruner
from .prompt_encoding import PromptEncoding


class Sam3ConceptTrackingSession(ConceptTrackingSession):
    """
    Concept tracking session of SAM 3 video, backed by a ``transformers`` inference session.

    Every query of the prompt is a SAM 3 video prompt whose features come from ``PromptEncoder``, so visual queries
    are tracked like phrases; each track reports the query that detected it. ``transformers`` names a prompt by its
    text, so every query is registered under its own key (``QUERY_KEY_FORMAT``) with its features already encoded. Instances
    below ``confidence_threshold`` (detection score of the track) or ``tracker_score_threshold`` (presence on the
    frame) are hidden but keep being tracked. A backward pass starts on its start frame like a forward pass, although
    ``transformers`` starts it one frame earlier; after a forward pass it skips the start frame already tracked. A
    reset restores the track numbering of a new session, so a repeated propagation yields the same track ids. A
    removed track is also hidden on the frames the hot start still holds back.

    SAM 3 never removes tracks of a stream, so a stream drops the tracks without a visible instance for
    ``stream_lost_track_frame_count`` frames, and frees the tracker history (``HistoryPruner``) and the per-frame
    fusion records (``FusionStatePruner``) its memory never reads again after every frame.
    """

    MASKS_KEY: ClassVar[str] = "masks"
    SCORES_KEY: ClassVar[str] = "scores"
    OBJECT_IDS_KEY: ClassVar[str] = "object_ids"
    PROMPT_TO_OBJECT_IDS_KEY: ClassVar[str] = "prompt_to_obj_ids"
    INITIAL_MAX_OBJECT_ID: ClassVar[int] = -1
    QUERY_KEY_FORMAT: ClassVar[str] = "query {query_id}"

    def __init__(
        self,
        model: Sam3VideoModel,
        processor: Sam3VideoProcessor,
        inference_session: Sam3VideoInferenceSession,
        runtime: TorchRuntime,
        prompt_encoding: PromptEncoding,
        settings: ConceptTrackerSettings,
        image_size: ImageSize | None,
        frame_count: int,
    ) -> None:
        """
        Parameters
        ----------
        model : Sam3VideoModel
            Loaded SAM 3 video model in evaluation mode.
        processor : Sam3VideoProcessor
            Processor of the model.
        inference_session : Sam3VideoInferenceSession
            ``transformers`` session, holding a whole video or empty for a stream.
        runtime : TorchRuntime
            Device and dtype of the model.
        prompt_encoding : PromptEncoding
            Tracked classes and the encoded features of their queries.
        settings : ConceptTrackerSettings
            Score thresholds and stream limits.
        image_size : ImageSize | None
            Size of the frames; ``None`` for a stream.
        frame_count : int
            Number of frames of a whole video; ``0`` for a stream.
        """
        super().__init__(
            prompt=prompt_encoding.prompt,
            image_size=image_size,
            frame_count=frame_count,
            is_streaming=image_size is None,
        )
        self._model: Sam3VideoModel = model
        self._processor: Sam3VideoProcessor = processor
        self._inference_session: Sam3VideoInferenceSession = inference_session
        self._runtime: TorchRuntime = runtime
        self._settings: ConceptTrackerSettings = settings
        self._prompt_encoding: PromptEncoding = prompt_encoding
        self._query_ids_by_key: dict[str, int] = {
            self.QUERY_KEY_FORMAT.format(query_id=query_id): query_id
            for query_id in range(len(prompt_encoding.prompt.queries))
        }
        self._last_seen_frame_by_track: dict[int, int] = {}
        self._history_pruner: HistoryPruner = HistoryPruner(
            history_length=self._history_length_of(model),
            conditioning_frame_limit=settings.stream_conditioning_frame_limit,
        )
        self._fusion_state_pruner: FusionStatePruner = FusionStatePruner()
        self._add_query_prompts()

    @property
    def track_ids(self) -> tuple[int, ...]:
        """
        Identifiers of the current tracks.

        Returns
        -------
        tuple[int, ...]
            Track identifiers known to the model.
        """
        object_ids: list[int] = cast(list[int], self._inference_session.obj_ids)
        return tuple(int(object_id) for object_id in object_ids)

    def _propagate(
        self, is_reverse: bool, is_start_frame_tracked: bool, start_frame_index: int, max_frame_count: int | None
    ) -> Iterator[ConceptFrame]:
        is_start_frame_shifted: bool = is_reverse and not is_start_frame_tracked
        model_start_frame_index: int = start_frame_index + 1 if is_start_frame_shifted else start_frame_index
        model_max_frame_count: int | None = (
            max_frame_count + 1 if is_start_frame_shifted and max_frame_count is not None else max_frame_count
        )
        for output in self._model.propagate_in_video_iterator(
            self._inference_session,
            start_frame_idx=model_start_frame_index,
            max_frame_num_to_track=model_max_frame_count,
            reverse=is_reverse,
        ):
            yield self._concept_frame_of(output)

    def _track_next(self, frame: Image.Image, frame_index: int) -> ConceptFrame:
        inputs: BatchEncoding = self._processor(images=frame, return_tensors="pt")
        pixel_values: torch.Tensor = self._runtime.to_model_input(inputs["pixel_values"][0])
        with torch.inference_mode():
            output: Sam3VideoSegmentationOutput = self._model(
                inference_session=self._inference_session, frame_idx=frame_index, frame=pixel_values
            )
        concept_frame: ConceptFrame = self._concept_frame_of(output)
        self._drop_lost_tracks(frame_index)
        output_dicts: dict[int, dict[str, dict[int, object]]] = self._inference_session.output_dict_per_obj
        tracked_frames_per_object: dict[int, dict[int, object]] = self._inference_session.frames_tracked_per_obj
        point_inputs: dict[int, dict[int, object]] = self._inference_session.point_inputs_per_obj
        mask_inputs: dict[int, dict[int, object]] = self._inference_session.mask_inputs_per_obj
        self._history_pruner.prune(
            frame_index,
            self._inference_session.processed_frames,
            output_dicts,
            tracked_frames_per_object,
            (point_inputs, mask_inputs),
        )
        self._fusion_state_pruner.prune(self._inference_session, frame_index)
        return concept_frame

    def _reset(self) -> None:
        self._inference_session.reset_state()
        self._inference_session.max_obj_id = self.INITIAL_MAX_OBJECT_ID
        self._inference_session.hotstart_removed_obj_ids.clear()
        self._last_seen_frame_by_track.clear()
        self._add_query_prompts()

    def _remove_track(self, track_id: int) -> None:
        self._inference_session.remove_object(track_id, strict=True)
        self._last_seen_frame_by_track.pop(track_id, None)
        if self._is_streaming:
            self._fusion_state_pruner.forget_track(self._inference_session, track_id)
        else:
            self._inference_session.hotstart_removed_obj_ids.add(track_id)

    def _add_query_prompts(self) -> None:
        for key, query_id in self._query_ids_by_key.items():
            prompt_id: int = self._inference_session.add_prompt(key)
            self._inference_session.prompt_embeddings[prompt_id] = self._prompt_encoding.text_embeddings(query_id, 1)
            self._inference_session.prompt_attention_masks[prompt_id] = self._prompt_encoding.query_attention_mask(
                query_id, 1
            )

    def _drop_lost_tracks(self, frame_index: int) -> None:
        for track_id in self.track_ids:
            last_seen_frame_index: int = self._last_seen_frame_by_track.setdefault(track_id, frame_index)
            if frame_index - last_seen_frame_index >= self._settings.stream_lost_track_frame_count:
                self._remove_track(track_id)

    def _concept_frame_of(self, output: Sam3VideoSegmentationOutput) -> ConceptFrame:
        image_size: ImageSize = self._require_image_size()
        frame_index: int = int(output.frame_idx if output.frame_idx is not None else 0)
        processed: Mapping[str, object] = cast(
            Mapping[str, object],
            self._processor.postprocess_outputs(
                self._inference_session, output, original_sizes=[[image_size.height, image_size.width]]
            ),
        )
        object_ids: IntArray = self._tensor_of(processed, self.OBJECT_IDS_KEY).cpu().numpy().astype(np.int64)
        if not len(object_ids):
            return ConceptFrame.empty(frame_index, self._prompt, image_size)
        scores: FloatArray = self._tensor_of(processed, self.SCORES_KEY).float().cpu().numpy().astype(np.float64)
        masks: BoolArray = self._tensor_of(processed, self.MASKS_KEY).bool().cpu().numpy()
        query_ids: IntArray = self._query_ids_of(processed, object_ids)
        tracker_scores: FloatArray = self._tracker_scores_of(output, object_ids, scores)
        is_visible: BoolArray = masks.any(axis=(1, 2))
        for track_id in object_ids[is_visible].tolist():
            self._last_seen_frame_by_track[int(track_id)] = frame_index
        is_kept: BoolArray = (
            (scores >= self._settings.confidence_threshold)
            & (tracker_scores >= self._settings.tracker_score_threshold)
            & is_visible
        )
        return ConceptFrame(
            frame_index=frame_index,
            instances=SegmentationResult(
                masks=masks[is_kept],
                confidences=np.clip(scores[is_kept], 0.0, 1.0),
                query_ids=query_ids[is_kept],
                prompt=self._prompt,
                image_size=image_size,
            ),
            track_ids=object_ids[is_kept],
            tracker_scores=tracker_scores[is_kept],
        )

    def _query_ids_of(self, processed: Mapping[str, object], object_ids: IntArray) -> IntArray:
        raw_mapping: object = processed[self.PROMPT_TO_OBJECT_IDS_KEY]
        if not isinstance(raw_mapping, dict):
            raise TypeError(f"unexpected prompt mapping: {type(raw_mapping)}")
        prompt_to_object_ids: dict[str, list[int]] = cast(dict[str, list[int]], raw_mapping)
        query_id_by_object: dict[int, int] = {}
        for key, key_object_ids in prompt_to_object_ids.items():
            for object_id in key_object_ids:
                query_id_by_object[int(object_id)] = self._query_ids_by_key[key]
        return np.array([query_id_by_object[int(object_id)] for object_id in object_ids], dtype=np.int64)

    @staticmethod
    def _tracker_scores_of(
        output: Sam3VideoSegmentationOutput, object_ids: IntArray, detection_scores: FloatArray
    ) -> FloatArray:
        tracker_score_by_object: dict[int, float] = output.obj_id_to_tracker_score or {}
        tracker_scores: FloatArray = np.array(
            [
                tracker_score_by_object.get(int(object_id), float(detection_score))
                for object_id, detection_score in zip(object_ids.tolist(), detection_scores.tolist(), strict=True)
            ],
            dtype=np.float64,
        )
        return np.clip(tracker_scores, 0.0, 1.0)

    @staticmethod
    def _history_length_of(model: Sam3VideoModel) -> int:
        tracker_config: object = cast(object, model.config.tracker_config)
        memory_frame_count: object = getattr(tracker_config, "num_maskmem", None)
        pointer_frame_count: object = getattr(tracker_config, "max_object_pointers_in_encoder", None)
        if not isinstance(memory_frame_count, int) or not isinstance(pointer_frame_count, int):
            raise TypeError(f"unexpected tracker configuration: {type(tracker_config)}")
        return max(memory_frame_count, pointer_frame_count)

    def _require_image_size(self) -> ImageSize:
        if self._image_size is None:
            raise ValueError("the session has no frame yet.")
        return self._image_size

    @staticmethod
    def _tensor_of(processed: Mapping[str, object], key: str) -> torch.Tensor:
        value: object = processed[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"expected a tensor for {key!r}. got {type(value)}")
        return value
