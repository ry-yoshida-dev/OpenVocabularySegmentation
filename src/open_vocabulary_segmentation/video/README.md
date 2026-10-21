# video

## Overview

Video segmentation with a memory of past frames. Two kinds of tracking share the same session pattern: an offline
session over a whole video (prompt any frame, propagate forward, backward or both) and a streaming session fed one
frame at a time with bounded memory, e.g. from a camera.

| Need | Tracker | Backends | Input |
| ---- | ------- | -------- | ----- |
| Follow objects you mark | `ObjectTracker` | SAM 2 / SAM 2.1, SAM 3 tracker | `ObjectPrompt` per object id on any frame |
| Find and follow every instance of a concept | `ConceptTracker` | SAM 3 video | `Prompt` of text or visual queries |

## Components

| Component | Description |
| --------- | ----------- |
| [object_tracking/](./object_tracking/README.md) | `ObjectTracker`, `TrackingSession` and their results `TrackedFrame` and `TrackedObject`. |
| [concept_tracking/](./concept_tracking/README.md) | `ConceptTracker`, `ConceptTrackingSession` and their result `ConceptFrame`. |

## Examples

```python
session = (
    ObjectTrackerSettings(backend=TrackerBackend.SAM2, weights_path="facebook/sam2.1-hiera-tiny")
    .build()
    .start_session(frames)
)
session.prompt_object(0, 1, ObjectPrompt.from_coordinates(positive_points=[(420, 310)]))
for frame in session.propagate():
    print(frame.frame_index, frame.object(1).area)
```
