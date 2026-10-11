"""
Track objects prompted on one frame through a video file and write a video of the tracked masks.

Usage
-----
python examples/track_video.py VIDEO OUTPUT.mp4 --backend sam2 --weights facebook/sam2.1-hiera-tiny \
    --frame 0 --object "1:point=420,310" --object "2:box=50,60,300,400"

Each ``--object`` is ``id:elements`` with any number of ``point=x,y`` and ``negative=x,y`` and at most one
``box=x_min,y_min,x_max,y_max``, drawn on ``--frame``. By default the whole video is loaded and the objects are
propagated forward and backward from that frame; ``--stream`` reads the video frame by frame instead, prompting on
``--frame`` when it arrives, with bounded memory.
"""

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from image_container import ArrayImageContainer, ChannelOrder, ImageContainer, ImageContainerType

from open_vocabulary_segmentation import (
    Device,
    MaskOverlayRenderer,
    ObjectPrompt,
    ObjectTracker,
    ObjectTrackerSettings,
    Precision,
    PropagationDirection,
    TrackedFrame,
    TrackerBackend,
    TrackingSession,
)

OBJECT_SEPARATOR: str = ":"
ELEMENT_SEPARATOR: str = "="
COORDINATE_SEPARATOR: str = ","
VIDEO_CODEC: str = "mp4v"


@dataclass(frozen=True)
class PromptedObject:
    """
    Object prompt of one ``--object`` argument.

    Attributes
    ----------
    object_id : int
        Identifier of the object.
    prompt : ObjectPrompt
        Points and box of the object.
    """

    object_id: int
    prompt: ObjectPrompt


def read_frames(path: Path) -> Iterator[ImageContainerType]:
    """
    Read the frames of a video file one by one.

    Parameters
    ----------
    path : Path
        Video file readable by OpenCV.

    Raises
    ------
    FileNotFoundError
        If the video cannot be opened.

    Yields
    ------
    ImageContainerType
        BGR frames in order.
    """
    capture: cv2.VideoCapture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    try:
        while True:
            is_read, frame = capture.read()
            if not is_read:
                return
            yield ImageContainer.register(np.ascontiguousarray(frame), ChannelOrder.BGR)
    finally:
        capture.release()


def frame_rate_of(path: Path) -> float:
    """
    Frame rate of a video file.

    Parameters
    ----------
    path : Path
        Video file readable by OpenCV.

    Returns
    -------
    float
        Frames per second; 30 when the file does not tell.
    """
    capture: cv2.VideoCapture = cv2.VideoCapture(str(path))
    frame_rate: float = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return frame_rate if frame_rate > 0 else 30.0


def parse_object(description: str) -> PromptedObject:
    """
    Read one ``id:point=x,y negative=x,y box=x_min,y_min,x_max,y_max`` argument.

    Parameters
    ----------
    description : str
        Object id, a colon, then space-separated prompt elements.

    Raises
    ------
    ValueError
        If an element is unknown or malformed.

    Returns
    -------
    PromptedObject
        Object id and its prompt.
    """
    object_id, _, elements = description.partition(OBJECT_SEPARATOR)
    positive_points: list[tuple[float, float]] = []
    negative_points: list[tuple[float, float]] = []
    box_xyxy: tuple[float, float, float, float] | None = None
    for element in elements.split():
        key, _, value = element.partition(ELEMENT_SEPARATOR)
        numbers: list[float] = [float(number) for number in value.split(COORDINATE_SEPARATOR)]
        match key, len(numbers):
            case "point", 2:
                positive_points.append((numbers[0], numbers[1]))
            case "negative", 2:
                negative_points.append((numbers[0], numbers[1]))
            case "box", 4:
                box_xyxy = (numbers[0], numbers[1], numbers[2], numbers[3])
            case _:
                raise ValueError(f"malformed prompt element {element!r}")
    return PromptedObject(
        object_id=int(object_id),
        prompt=ObjectPrompt.from_coordinates(
            positive_points=positive_points, negative_points=negative_points, box_xyxy=box_xyxy
        ),
    )


def write_video(path: Path, frames: list[ArrayImageContainer], frame_rate: float) -> None:
    """
    Write RGB renderings as a video file.

    Parameters
    ----------
    path : Path
        Output video file.
    frames : list[ArrayImageContainer]
        Rendered frames in order, all of one size.
    frame_rate : float
        Frames per second.
    """
    if not frames:
        return
    height, width = frames[0].value.shape[:2]
    writer: cv2.VideoWriter = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*VIDEO_CODEC), frame_rate, (width, height)
    )
    for frame in frames:
        writer.write(frame.to_array(ChannelOrder.BGR))
    writer.release()


def main() -> None:
    """
    Parse arguments, track the prompted objects and write the overlay video.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", type=TrackerBackend, choices=list(TrackerBackend), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--object", dest="objects", action="append", required=True)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--device", type=Device, choices=list(Device), default=Device.AUTO)
    parser.add_argument("--precision", type=Precision, choices=list(Precision), default=Precision.FLOAT32)
    arguments: argparse.Namespace = parser.parse_args()

    tracker: ObjectTracker = ObjectTrackerSettings(
        backend=arguments.backend,
        weights_path=arguments.weights,
        device=arguments.device,
        precision=arguments.precision,
    ).build()
    prompted_objects: list[PromptedObject] = [parse_object(description) for description in arguments.objects]
    prompts: dict[int, ObjectPrompt] = {prompted.object_id: prompted.prompt for prompted in prompted_objects}
    renderer: MaskOverlayRenderer = MaskOverlayRenderer()
    rendered: dict[int, ArrayImageContainer] = {}

    if arguments.stream:
        session: TrackingSession = tracker.start_stream()
        for frame_index, frame in enumerate(read_frames(arguments.video)):
            tracked: TrackedFrame = session.track_next(frame, prompts if frame_index == arguments.frame else None)
            rendered[frame_index] = renderer.render_tracked(frame, tracked)
            print(f"frame {frame_index}: objects {tracked.visible_objects().object_ids.tolist()}")
    else:
        frames: list[ImageContainerType] = list(read_frames(arguments.video))
        session = tracker.start_session(frames)
        session.prompt_objects(arguments.frame, prompts)
        for tracked in session.propagate(PropagationDirection.BIDIRECTIONAL, start_frame_index=arguments.frame):
            rendered[tracked.frame_index] = renderer.render_tracked(frames[tracked.frame_index], tracked)
            print(f"frame {tracked.frame_index}: objects {tracked.visible_objects().object_ids.tolist()}")

    write_video(arguments.output, [rendered[index] for index in sorted(rendered)], frame_rate_of(arguments.video))


if __name__ == "__main__":
    main()
