from collections.abc import Iterator, Mapping, Sequence
from typing import ClassVar

import numpy as np
import pytest
import torch
from image_container import BinaryImage
from PIL import Image

from open_vocabulary_segmentation import (
    ConceptFrame,
    ConceptTracker,
    ConceptTrackerSettings,
    ConceptTrackingParameters,
    ConceptTrackingSession,
    Device,
    ImageSize,
    MaskLogits,
    ObjectPrompt,
    ObjectTracker,
    ObjectTrackerSettings,
    Prompt,
    PromptableBackend,
    PropagationDirection,
    SegmentationResult,
    TrackedFrame,
    TrackerBackend,
    TrackingSession,
)
from open_vocabulary_segmentation.array_types import BoolArray
from open_vocabulary_segmentation.backends.memory_tracker.history_pruner import HistoryPruner

FRAME_SIZE: ImageSize = ImageSize(width=20, height=10)


def build_frame(frame_index: int, object_ids: Sequence[int]) -> TrackedFrame:
    masks: BoolArray = np.zeros((len(object_ids), FRAME_SIZE.height, FRAME_SIZE.width), dtype=np.bool_)
    for index in range(len(object_ids)):
        masks[index, 0:5, index * 5 : index * 5 + 5] = True
    return TrackedFrame(
        frame_index=frame_index,
        object_ids=np.array(list(object_ids), dtype=np.int64),
        masks=masks,
        presence_scores=np.full(len(object_ids), 0.9),
        image_size=FRAME_SIZE,
    )


class StubTrackingSession(TrackingSession):
    def __init__(self, image_size: ImageSize | None, frame_count: int) -> None:
        super().__init__(image_size=image_size, frame_count=frame_count, is_streaming=image_size is None)
        self.prompted: list[tuple[int, list[int]]] = []
        self.passes: list[tuple[bool, int | None, int | None]] = []
        self.streamed_prompt_ids: list[list[int]] = []
        self.reset_count: int = 0

    def _prompt(self, frame_index: int, prompts: Mapping[int, ObjectPrompt]) -> TrackedFrame:
        self.prompted.append((frame_index, list(prompts)))
        return build_frame(frame_index, self.object_ids)

    def _propagate(
        self, is_reverse: bool, start_frame_index: int | None, max_frame_count: int | None
    ) -> Iterator[TrackedFrame]:
        self.passes.append((is_reverse, start_frame_index, max_frame_count))
        start: int = 1 if start_frame_index is None else start_frame_index
        indices: range = range(start, -1, -1) if is_reverse else range(start, self.frame_count)
        for frame_index in indices:
            yield build_frame(frame_index, self.object_ids)

    def _track_next(self, frame: Image.Image, frame_index: int, prompts: Mapping[int, ObjectPrompt]) -> TrackedFrame:
        self.streamed_prompt_ids.append(list(prompts))
        return build_frame(frame_index, self.object_ids)

    def _reset(self) -> None:
        self.reset_count += 1


class StubObjectTracker(ObjectTracker):
    BACKENDS: ClassVar[frozenset[TrackerBackend]] = frozenset({TrackerBackend.SAM2})

    def _start_offline(self, frames: Sequence[Image.Image]) -> TrackingSession:
        return StubTrackingSession(ImageSize.from_image(frames[0]), len(frames))

    def _start_streaming(self) -> TrackingSession:
        return StubTrackingSession(None, 0)


def build_tracker() -> StubObjectTracker:
    return StubObjectTracker(ObjectTrackerSettings(backend=TrackerBackend.SAM2, weights_path="stub", device=Device.CPU))


def frames(count: int) -> list[Image.Image]:
    return [Image.new("RGB", (FRAME_SIZE.width, FRAME_SIZE.height)) for _ in range(count)]


def test_tracked_frame_tools() -> None:
    frame: TrackedFrame = build_frame(3, [7, 2])
    assert 7 in frame and 5 not in frame
    second_box = frame.object(2).box
    assert second_box is not None
    assert second_box.value.tolist() == [5, 0, 10, 5]
    assert frame.object(7).is_visible
    assert int(frame.object_map()[0, 6]) == 2
    assert int(frame.union_mask().value.sum()) == 50
    with pytest.raises(KeyError):
        frame.object(5)
    hidden: TrackedFrame = TrackedFrame(
        frame_index=0,
        object_ids=np.array([1]),
        masks=np.zeros((1, 10, 20), dtype=np.bool_),
        presence_scores=np.array([0.1]),
        image_size=FRAME_SIZE,
    )
    assert len(hidden.visible_objects()) == 0
    assert hidden.object(1).box is None
    with pytest.raises(ValueError, match="unique"):
        build_frame(0, [1, 1])


def test_offline_session_prompts_and_propagates_in_both_directions() -> None:
    session: TrackingSession = build_tracker().start_session(frames(4))
    assert not session.is_streaming and session.frame_count == 4 and session.image_size == FRAME_SIZE
    first: TrackedFrame = session.prompt_objects(
        1,
        {
            3: ObjectPrompt.from_coordinates(positive_points=[(2, 2)]),
            9: ObjectPrompt.from_coordinates(box_xyxy=(0, 0, 5, 5)),
        },
    )
    assert first.object_ids.tolist() == [3, 9]
    session.prompt_object(2, 3, ObjectPrompt(mask=BinaryImage(value=np.ones((10, 20), dtype=np.bool_))))
    assert session.object_ids == (3, 9)
    propagated: list[int] = [frame.frame_index for frame in session.propagate(PropagationDirection.BIDIRECTIONAL)]
    assert propagated == [1, 2, 3, 0]
    assert isinstance(session, StubTrackingSession)
    assert [is_reverse for is_reverse, _, _ in session.passes] == [False, True]
    session.reset()
    assert len(session.object_ids) == 0 and session.reset_count == 1


def test_offline_session_rejects_invalid_requests() -> None:
    session: TrackingSession = build_tracker().start_session(frames(2))
    with pytest.raises(ValueError, match="prompt at least one object"):
        list(session.propagate())
    with pytest.raises(ValueError, match="frame_index must be in"):
        session.prompt_object(2, 0, ObjectPrompt.from_coordinates(box_xyxy=(0, 0, 4, 4)))
    with pytest.raises(ValueError, match="must not be negative"):
        session.prompt_object(0, -1, ObjectPrompt.from_coordinates(box_xyxy=(0, 0, 4, 4)))
    with pytest.raises(ValueError, match="cannot be combined"):
        session.prompt_object(
            0,
            0,
            ObjectPrompt.from_coordinates(
                positive_points=[(1, 1)], mask=BinaryImage(value=np.ones((10, 20), dtype=np.bool_))
            ),
        )
    logits: MaskLogits = MaskLogits(
        logits=np.zeros((4, 4), dtype=np.float32),
        backend=PromptableBackend.SAM2,
        weights_path="w",
        image_size=FRAME_SIZE,
    )
    with pytest.raises(ValueError, match="take binary masks"):
        session.prompt_object(0, 0, ObjectPrompt(mask=logits))
    with pytest.raises(ValueError, match="inside the 20x10 image"):
        session.prompt_object(0, 0, ObjectPrompt.from_coordinates(positive_points=[(25, 1)]))
    with pytest.raises(ValueError, match="offline session"):
        session.track_next(frames(1)[0])
    with pytest.raises(ValueError, match="same size"):
        build_tracker().start_session([Image.new("RGB", (20, 10)), Image.new("RGB", (10, 10))])
    with pytest.raises(ValueError, match="at least one frame"):
        build_tracker().start_session([])


def test_streaming_session_tracks_frame_by_frame() -> None:
    session: TrackingSession = build_tracker().start_stream()
    assert session.is_streaming and session.image_size is None
    session.track_next(frames(1)[0])
    tracked: TrackedFrame = session.track_next(
        frames(1)[0], {4: ObjectPrompt.from_coordinates(positive_points=[(1, 1)])}
    )
    assert tracked.frame_index == 1
    assert session.frame_count == 2 and session.object_ids == (4,)
    with pytest.raises(ValueError, match="stream frames must be 20x10"):
        session.track_next(Image.new("RGB", (8, 8)))
    with pytest.raises(ValueError, match="streaming session"):
        session.prompt_object(0, 1, ObjectPrompt.from_coordinates(positive_points=[(1, 1)]))
    with pytest.raises(ValueError, match="streaming session"):
        list(session.propagate())


def test_tracker_settings_must_match_the_backend() -> None:
    with pytest.raises(ValueError, match=r"needs settings of \['sam2'\]"):
        StubObjectTracker(ObjectTrackerSettings(backend=TrackerBackend.SAM3, weights_path="stub"))


def build_concept_frame(frame_index: int, prompt: Prompt, track_ids: Sequence[int]) -> ConceptFrame:
    masks: BoolArray = np.zeros((len(track_ids), FRAME_SIZE.height, FRAME_SIZE.width), dtype=np.bool_)
    for index in range(len(track_ids)):
        masks[index, 5:10, index * 4 : index * 4 + 4] = True
    return ConceptFrame(
        frame_index=frame_index,
        instances=SegmentationResult(
            masks=masks,
            confidences=np.linspace(0.9, 0.5, len(track_ids)),
            query_ids=np.arange(len(track_ids), dtype=np.int64) % len(prompt.queries),
            prompt=prompt,
            image_size=FRAME_SIZE,
        ),
        track_ids=np.array(list(track_ids), dtype=np.int64),
        tracker_scores=np.full(len(track_ids), 0.8),
    )


class StubConceptSession(ConceptTrackingSession):
    def __init__(self, prompt: Prompt, image_size: ImageSize | None, frame_count: int) -> None:
        super().__init__(prompt=prompt, image_size=image_size, frame_count=frame_count, is_streaming=image_size is None)
        self.removed: list[int] = []
        self.reset_count: int = 0
        self.is_start_frame_tracked_per_pass: list[bool] = []

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(track_id for track_id in (0, 1) if track_id not in self.removed)

    def _propagate(
        self, is_reverse: bool, is_start_frame_tracked: bool, start_frame_index: int, max_frame_count: int | None
    ) -> Iterator[ConceptFrame]:
        self.is_start_frame_tracked_per_pass.append(is_start_frame_tracked)
        indices: range = range(start_frame_index, -1, -1) if is_reverse else range(start_frame_index, self.frame_count)
        for frame_index in indices:
            yield build_concept_frame(frame_index, self.prompt, self.track_ids)

    def _track_next(self, frame: Image.Image, frame_index: int) -> ConceptFrame:
        return build_concept_frame(frame_index, self.prompt, self.track_ids)

    def _remove_track(self, track_id: int) -> None:
        self.removed.append(track_id)

    def _reset(self) -> None:
        self.reset_count += 1
        self.removed.clear()


class StubConceptTracker(ConceptTracker):
    def _start_offline(self, frames: Sequence[Image.Image], prompt: Prompt) -> ConceptTrackingSession:
        return StubConceptSession(prompt, ImageSize.from_image(frames[0]), len(frames))

    def _start_streaming(self, prompt: Prompt) -> ConceptTrackingSession:
        return StubConceptSession(prompt, None, 0)


def test_concept_frame_keeps_tracks_aligned() -> None:
    prompt: Prompt = Prompt.from_class_names(("car", "dog"))
    frame: ConceptFrame = build_concept_frame(2, prompt, [5, 8])
    assert [tracked.track_id for tracked in frame] == [5, 8]
    assert [tracked.tracker_score for tracked in frame] == [0.8, 0.8]
    assert frame.filter_by_class("dog").tracker_scores.tolist() == [0.8]
    assert frame.segmentation_of(8).class_name == "dog"
    assert frame.filter_by_class("dog").track_ids.tolist() == [8]
    assert int(frame.track_map()[6, 1]) == 5 and int(frame.track_map()[0, 0]) == -1
    assert 5 in frame
    with pytest.raises(KeyError):
        frame.segmentation_of(1)
    assert len(ConceptFrame.empty(0, prompt, FRAME_SIZE)) == 0


def test_concept_sessions_validate_and_remove_tracks() -> None:
    tracker: StubConceptTracker = StubConceptTracker(ConceptTrackerSettings(weights_path="stub"))
    prompt: Prompt = Prompt.from_class_names(("car", "dog"))
    session: StubConceptSession = StubConceptSession(prompt, FRAME_SIZE, 3)
    assert [frame.frame_index for frame in session.propagate(PropagationDirection.BIDIRECTIONAL, 1)] == [1, 2, 0]
    assert session.is_start_frame_tracked_per_pass == [False, True]
    assert session.reset_count == 0
    session.remove_track(0)
    assert session.track_ids == (1,)
    with pytest.raises(KeyError):
        session.remove_track(0)
    assert len(list(session.propagate())) == 3
    assert session.reset_count == 1
    assert set(session.track_ids) == {0, 1}
    session.reset()
    assert session.reset_count == 2
    with pytest.raises(ValueError, match="start_frame_index"):
        list(session.propagate(start_frame_index=3))
    stream: ConceptTrackingSession = tracker.start_stream(prompt)
    assert stream.track_next(frames(1)[0]).frame_index == 0
    assert stream.track_next(frames(1)[0]).frame_index == 1
    with pytest.raises(ValueError, match="offline session"):
        session.track_next(frames(1)[0])


def test_concept_tracking_parameters_validate() -> None:
    assert ConceptTrackingParameters().detection_threshold is None
    with pytest.raises(ValueError, match="detection_threshold"):
        ConceptTrackingParameters(detection_threshold=1.5)
    with pytest.raises(ValueError, match="hotstart_delay"):
        ConceptTrackingParameters(hotstart_delay=-1)
    with pytest.raises(ValueError, match="hotstart_unmatched_frame_count must not exceed"):
        ConceptTrackingParameters(hotstart_delay=4, hotstart_unmatched_frame_count=5)
    assert ConceptTrackingParameters(hotstart_delay=4, hotstart_duplicate_frame_count=4).hotstart_delay == 4
    with pytest.raises(ValueError, match="tracker_score_threshold"):
        ConceptTrackerSettings(weights_path="stub", tracker_score_threshold=2.0)
    with pytest.raises(ValueError, match="stream_lost_track_frame_count"):
        ConceptTrackerSettings(weights_path="stub", stream_lost_track_frame_count=0)


def test_propagation_direction_passes() -> None:
    assert PropagationDirection.FORWARD.is_reverse_per_pass == (False,)
    assert PropagationDirection.BACKWARD.is_reverse_per_pass == (True,)
    assert PropagationDirection.BIDIRECTIONAL.is_reverse_per_pass == (False, True)


def test_history_pruner_bounds_a_stream() -> None:
    pruner: HistoryPruner = HistoryPruner(history_length=2, conditioning_frame_limit=3)
    processed_frames: dict[int, torch.Tensor] = {index: torch.ones(2) for index in range(10)}
    outputs: dict[int, dict[str, dict[int, object]]] = {
        0: {
            HistoryPruner.CONDITIONING_OUTPUTS_KEY: {index: index for index in (0, 3, 5, 8, 9)},
            HistoryPruner.NON_CONDITIONING_OUTPUTS_KEY: {index: index for index in range(10)},
        }
    }
    tracked_frames: dict[int, dict[int, object]] = {0: {index: index for index in range(10)}}
    point_inputs: dict[int, dict[int, object]] = {0: {3: "points", 9: "points"}}
    pruner.prune(9, processed_frames, outputs, tracked_frames, (point_inputs,))
    assert sorted(outputs[0][HistoryPruner.CONDITIONING_OUTPUTS_KEY]) == [0, 8, 9]
    assert sorted(outputs[0][HistoryPruner.NON_CONDITIONING_OUTPUTS_KEY]) == [7, 8, 9]
    assert sorted(tracked_frames[0]) == [7, 8, 9]
    assert list(point_inputs[0]) == [9]
    assert processed_frames[8].numel() == 0 and processed_frames[9].numel() == 2
    with pytest.raises(ValueError, match="conditioning_frame_limit"):
        HistoryPruner(history_length=2, conditioning_frame_limit=1)
    with pytest.raises(ValueError, match="stream_conditioning_frame_limit"):
        ObjectTrackerSettings(backend=TrackerBackend.SAM2, weights_path="stub", stream_conditioning_frame_limit=1)
