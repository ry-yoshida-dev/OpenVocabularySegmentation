"""
Detect and track every instance of text phrases through a video file with SAM 3 video and write the tracks.

Usage
-----
python examples/track_concepts.py VIDEO OUTPUT.mp4 --classes person "car:car,truck" --confidence-threshold 0.5

Every instance gets a track id kept across frames and is drawn in the color of its track. ``--stream`` reads the
video frame by frame with bounded memory instead of loading it whole.
"""

import argparse
from pathlib import Path

from image_container import ArrayImageContainer, ImageContainerType
from segment_directory import parse_prompt
from track_video import frame_rate_of, read_frames, write_video

from open_vocabulary_segmentation import (
    ConceptFrame,
    ConceptTracker,
    ConceptTrackerSettings,
    ConceptTrackingSession,
    Device,
    MaskOverlayRenderer,
    Precision,
    Prompt,
)


def describe(frame: ConceptFrame) -> str:
    """
    One line listing the tracks of a frame.

    Parameters
    ----------
    frame : ConceptFrame
        Tracked instances of a frame.

    Returns
    -------
    str
        ``track:class(confidence)`` entries.
    """
    return " ".join(
        f"{tracked.track_id}:{tracked.segmentation.class_name}({tracked.segmentation.confidence:.2f})"
        for tracked in frame
    )


def main() -> None:
    """
    Parse arguments, track the concepts and write the overlay video.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--weights", default="facebook/sam3")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--device", type=Device, choices=list(Device), default=Device.AUTO)
    parser.add_argument("--precision", type=Precision, choices=list(Precision), default=Precision.FLOAT32)
    arguments: argparse.Namespace = parser.parse_args()

    tracker: ConceptTracker = ConceptTrackerSettings(
        weights_path=arguments.weights,
        confidence_threshold=arguments.confidence_threshold,
        device=arguments.device,
        precision=arguments.precision,
    ).build()
    prompt: Prompt = parse_prompt(arguments.classes)
    renderer: MaskOverlayRenderer = MaskOverlayRenderer()
    rendered: dict[int, ArrayImageContainer] = {}

    if arguments.stream:
        session: ConceptTrackingSession = tracker.start_stream(prompt)
        for frame in read_frames(arguments.video):
            tracked: ConceptFrame = session.track_next(frame)
            rendered[tracked.frame_index] = renderer.render_concepts(frame, tracked)
            print(f"frame {tracked.frame_index}: {describe(tracked)}")
    else:
        frames: list[ImageContainerType] = list(read_frames(arguments.video))
        session = tracker.start_session(frames, prompt)
        for tracked in session.propagate():
            rendered[tracked.frame_index] = renderer.render_concepts(frames[tracked.frame_index], tracked)
            print(f"frame {tracked.frame_index}: {describe(tracked)}")

    write_video(arguments.output, [rendered[index] for index in sorted(rendered)], frame_rate_of(arguments.video))


if __name__ == "__main__":
    main()
