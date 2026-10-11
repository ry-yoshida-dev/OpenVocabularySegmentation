# object_tracking

## Overview

Promptable video segmentation: objects marked by an `ObjectPrompt` (points and/or a box, or a binary mask) on some
frames are segmented on every frame. Objects are identified by non-negative integers chosen by the caller; a prompt
on a frame adds an object or corrects it there, replacing its earlier prompt on that frame; corrections on frames
already tracked condition later propagations.

- **Offline** (`ObjectTracker.start_session(frames)`): prompt objects on any frames with `prompt_objects`, then
  `propagate` forward, backward or bidirectionally from a start frame. Frames are preprocessed and stored on the CPU
  unless `is_frame_storage_on_device`, and per-frame outputs are kept on the CPU unless `is_tracking_state_on_device`,
  so long videos fit in GPU memory.
- **Streaming** (`ObjectTracker.start_stream()`): `track_next(frame, prompts)` segments each new frame and takes the
  prompts drawn on it; frames before the first object are stored without inference. Only the memory the model reads
  is kept, with at most `stream_conditioning_frame_limit` prompted frames per object (the first and the latest), so
  the footprint stays bounded even when the stream is prompted again and again.

Each `TrackedFrame` holds the mask and the presence probability of every object; occluded objects have empty masks.

## Components

| Component | Description |
| --------- | ----------- |
| [tracker.py](./tracker.py) | `ObjectTracker`: base holding the model; validates frames and opens sessions. |
| [session.py](./session.py) | `TrackingSession`: prompt validation, propagation passes and streaming, independent of the model. |
| [settings.py](./settings.py) | `ObjectTrackerSettings`: backend, weights, device, precision, frame and state placement, stream limit; `build()` loads the tracker. |
| [tracked_frame.py](./tracked_frame.py) | `TrackedFrame`: masks and presence of every object on one frame; lookup by id, visible objects, object map. |
| [tracked_object.py](./tracked_object.py) | `TrackedObject`: one object on one frame with its box, area and polygons. |

## Examples

```python
tracker = ObjectTrackerSettings(backend=TrackerBackend.SAM3, weights_path="facebook/sam3").build()

session = tracker.start_session(frames)
session.prompt_objects(30, {1: ObjectPrompt.from_coordinates(box_xyxy=(50, 60, 300, 400)), 2: player_click})
masks = {
    frame.frame_index: frame for frame in session.propagate(PropagationDirection.BIDIRECTIONAL, start_frame_index=30)
}
session.prompt_object(80, 1, ObjectPrompt.from_coordinates(positive_points=[(210, 220)], negative_points=[(260, 240)]))

stream = tracker.start_stream()
for index, frame in enumerate(camera_frames):
    tracked = stream.track_next(frame, {1: first_click} if index == 0 else None)
```
