from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from PIL import Image

from ...image import ImageInput, RgbImage
from ...options import TrackerBackend
from .session import TrackingSession
from .settings import ObjectTrackerSettings


class ObjectTracker(ABC):
    """
    Common base of video object trackers: segments the objects prompted on some frames in every frame of a video.

    A tracker holds the model; each video is tracked in its own ``TrackingSession``. ``start_session`` takes a
    whole video, so objects may be prompted on any frame and propagated in both directions; ``start_stream`` takes
    frames one by one as they arrive, e.g. from a camera, with bounded memory.

    Attributes
    ----------
    settings : ObjectTrackerSettings
        Backend, weights, device and precision.
    """

    BACKENDS: ClassVar[frozenset[TrackerBackend]]

    def __init__(self, settings: ObjectTrackerSettings) -> None:
        """
        Parameters
        ----------
        settings : ObjectTrackerSettings
            Backend, weights, device and precision.

        Raises
        ------
        ValueError
            If the settings are for another backend.
        """
        if settings.backend not in self.BACKENDS:
            raise ValueError(
                f"{type(self).__name__} needs settings of {sorted(backend.value for backend in self.BACKENDS)}. "
                + f"got {settings.backend}"
            )
        self.settings: ObjectTrackerSettings = settings

    def start_session(self, frames: Sequence[ImageInput]) -> TrackingSession:
        """
        Start tracking in a whole video.

        Parameters
        ----------
        frames : Sequence[ImageInput]
            Frames of the video in order, all of one size.

        Raises
        ------
        ValueError
            If there is no frame or the frames differ in size.

        Returns
        -------
        TrackingSession
            Session over the video, without objects yet.
        """
        if not frames:
            raise ValueError("a video needs at least one frame.")
        rgb_frames: list[Image.Image] = [RgbImage.from_input(frame) for frame in frames]
        sizes: set[tuple[int, int]] = {frame.size for frame in rgb_frames}
        if len(sizes) != 1:
            raise ValueError(f"every frame must have the same size. got {sorted(sizes)}")
        return self._start_offline(rgb_frames)

    def start_stream(self) -> TrackingSession:
        """
        Start tracking frames that arrive one by one.

        Returns
        -------
        TrackingSession
            Streaming session without frames and objects; feed it with ``TrackingSession.track_next``.
        """
        return self._start_streaming()

    @abstractmethod
    def _start_offline(self, frames: Sequence[Image.Image]) -> TrackingSession:
        """
        Preprocess a whole video and open a session over it.

        Parameters
        ----------
        frames : Sequence[Image.Image]
            RGB frames of one size.

        Returns
        -------
        TrackingSession
            Offline session.
        """

    @abstractmethod
    def _start_streaming(self) -> TrackingSession:
        """
        Open a streaming session.

        Returns
        -------
        TrackingSession
            Streaming session.
        """
