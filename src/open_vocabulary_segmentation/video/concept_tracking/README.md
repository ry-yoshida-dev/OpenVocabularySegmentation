# concept_tracking

## Overview

Promptable concept segmentation in video (SAM 3 video): every instance of every query of a `Prompt` (text, and visual for SAM 3) is
detected on every frame, matched with the existing tracks, and given a track identifier when it first appears.
Each `ConceptFrame` is a regular `SegmentationResult` (class, matched query, mask and the detection score that
started the track as confidence) plus the aligned track ids and per-frame tracker (presence) scores; both scores have
thresholds in the settings. `ConceptTrackingParameters` override every threshold of the detection-tracking fusion:
detection, new-track, NMS and association thresholds, the hot start that holds back new tracks while duplicate and
spurious ones are removed, reconditioning, occlusion suppression, the track limit and the hole filling.

Like object tracking, a session is offline (`propagate`; every call starts from a fresh state, `reset` clears it) or
streaming (`track_next`). Streams drop tracks unseen for `stream_lost_track_frame_count` frames and keep a bounded
memory. `remove_track` stops a track and hides it on frames still held back; the detector may find the same instance
again under a new id.

## Components

| Component | Description |
| --------- | ----------- |
| [tracker.py](./tracker.py) | `ConceptTracker`: base holding the model; validates the prompt and frames and opens sessions. |
| [session.py](./session.py) | `ConceptTrackingSession`: propagation passes, streaming, reset and track removal, independent of the model. |
| [settings.py](./settings.py) | `ConceptTrackerSettings`: weights, score thresholds, parameters, device, precision, state placement and stream limits. |
| [parameters.py](./parameters.py) | `ConceptTrackingParameters`: overrides of the detection-tracking fusion; `None` keeps the checkpoint value. |
| [concept_frame.py](./concept_frame.py) | `ConceptFrame`: instances of one frame with their track ids and tracker scores; lookup by track, class filter, track map. |
| [tracked_segmentation.py](./tracked_segmentation.py) | `TrackedSegmentation`: one instance with its track id, frame index and tracker score. |

## Examples

```python
tracker = ConceptTrackerSettings(weights_path="facebook/sam3", confidence_threshold=0.5).build()
session = tracker.start_session(frames, Prompt.from_texts({"person": ("person",), "car": ("car", "truck")}))
for frame in session.propagate():
    for tracked in frame:
        print(frame.frame_index, tracked.track_id, tracked.segmentation.class_name, tracked.tracker_score)
```
