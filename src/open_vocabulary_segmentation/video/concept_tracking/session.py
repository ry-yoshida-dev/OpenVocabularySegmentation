from abc import ABC, abstractmethod
from collections.abc import Iterator

from open_vocabulary_detector import ImageSize, Prompt
from PIL import Image

from ...image import ImageInput, RgbImage
from ...options import PropagationDirection
from .concept_frame import ConceptFrame


class ConceptTrackingSession(ABC):
    """
    Tracking state of the concepts of one prompt in one video.

    Every instance of every query of the prompt is detected, given a track identifier on its first frame and
    tracked through the following frames; new instances join as they appear. An offline session
    (``ConceptTracker.start_session``) knows every frame and is processed with ``propagate``; every call starts
    from a fresh state, while the passes of one bidirectional call share their tracks. A streaming session
    (``ConceptTracker.start_stream``) receives frames one at a time through ``track_next``.
    """

    def __init__(self, prompt: Prompt, image_size: ImageSize | None, frame_count: int, is_streaming: bool) -> None:
        """
        Parameters
        ----------
        prompt : Prompt
            Tracked classes and their queries.
        image_size : ImageSize | None
            Size of the frames; ``None`` for a stream before its first frame.
        frame_count : int
            Number of frames known to the session.
        is_streaming : bool
            Whether frames arrive through ``track_next``.
        """
        self._prompt: Prompt = prompt
        self._image_size: ImageSize | None = image_size
        self._frame_count: int = frame_count
        self._is_streaming: bool = is_streaming
        self._is_propagated: bool = False

    @property
    def prompt(self) -> Prompt:
        """
        Tracked classes and their queries.

        Returns
        -------
        Prompt
            Prompt of the session.
        """
        return self._prompt

    @property
    def is_streaming(self) -> bool:
        """
        Whether frames arrive through ``track_next``.

        Returns
        -------
        bool
            True for a streaming session.
        """
        return self._is_streaming

    @property
    def frame_count(self) -> int:
        """
        Number of frames of the video, or frames received so far by a stream.

        Returns
        -------
        int
            Frame count.
        """
        return self._frame_count

    @property
    def image_size(self) -> ImageSize | None:
        """
        Size of the frames.

        Returns
        -------
        ImageSize | None
            Frame size; ``None`` for a stream before its first frame.
        """
        return self._image_size

    def propagate(
        self,
        direction: PropagationDirection = PropagationDirection.FORWARD,
        start_frame_index: int = 0,
        max_frame_count: int | None = None,
    ) -> Iterator[ConceptFrame]:
        """
        Detect and track the concepts through the video of an offline session.

        Parameters
        ----------
        direction : PropagationDirection, optional
            Forward, backward, or forward then backward from the start frame.
        start_frame_index : int, optional
            First frame of every pass.
        max_frame_count : int | None, optional
            Maximum number of frames per pass after the start frame; ``None`` runs to the end of the video.

        Raises
        ------
        ValueError
            If the session is streaming, the start frame is out of range or ``max_frame_count`` is negative.

        Yields
        ------
        ConceptFrame
            Instances with their tracks on each frame, in processing order; a frame reached by both passes of a
            bidirectional propagation is yielded once.
        """
        if self._is_streaming:
            raise ValueError("a streaming session tracks frames through track_next.")
        if not 0 <= start_frame_index < self._frame_count:
            raise ValueError(f"start_frame_index must be in [0, {self._frame_count}). got {start_frame_index}")
        if max_frame_count is not None and max_frame_count < 0:
            raise ValueError(f"max_frame_count must not be negative. got {max_frame_count}")
        if self._is_propagated:
            self._reset()
        self._is_propagated = True
        yielded_frame_indices: set[int] = set()
        for is_reverse in direction.is_reverse_per_pass:
            is_start_frame_tracked: bool = start_frame_index in yielded_frame_indices
            for frame in self._propagate(is_reverse, is_start_frame_tracked, start_frame_index, max_frame_count):
                if frame.frame_index in yielded_frame_indices:
                    continue
                yielded_frame_indices.add(frame.frame_index)
                yield frame

    def track_next(self, frame: ImageInput) -> ConceptFrame:
        """
        Detect and track the concepts on the next frame of a stream.

        Parameters
        ----------
        frame : ImageInput
            Next frame of the stream.

        Raises
        ------
        ValueError
            If the session is offline or the frame size differs from the previous frames.

        Returns
        -------
        ConceptFrame
            Instances with their tracks on the frame.
        """
        if not self._is_streaming:
            raise ValueError("an offline session knows its frames; use propagate.")
        rgb_frame: Image.Image = RgbImage.from_input(frame)
        frame_size: ImageSize = ImageSize.from_image(rgb_frame)
        if self._image_size is not None and frame_size != self._image_size:
            raise ValueError(
                f"stream frames must be {self._image_size.width}x{self._image_size.height}. "
                + f"got {frame_size.width}x{frame_size.height}"
            )
        self._image_size = frame_size
        tracked: ConceptFrame = self._track_next(rgb_frame, self._frame_count)
        self._frame_count += 1
        return tracked

    def reset(self) -> None:
        """
        Forget every track; an offline session keeps its frames and a stream continues with fresh tracks.
        """
        self._is_propagated = False
        self._reset()

    def remove_track(self, track_id: int) -> None:
        """
        Stop tracking one instance and hide it on the frames still to be yielded; its identifier is not reused.

        The detector still runs on every frame, so an instance it keeps detecting comes back as a new track; raise
        the confidence thresholds to suppress false detections.

        Parameters
        ----------
        track_id : int
            Track identifier.

        Raises
        ------
        KeyError
            If no track has ``track_id``.
        """
        if track_id not in self.track_ids:
            raise KeyError(f"track {track_id} does not exist. tracks: {list(self.track_ids)}")
        self._remove_track(track_id)

    @property
    @abstractmethod
    def track_ids(self) -> tuple[int, ...]:
        """
        Identifiers of the current tracks.

        Returns
        -------
        tuple[int, ...]
            Track identifiers.
        """

    @abstractmethod
    def _propagate(
        self, is_reverse: bool, is_start_frame_tracked: bool, start_frame_index: int, max_frame_count: int | None
    ) -> Iterator[ConceptFrame]:
        """
        Run one propagation pass.

        Parameters
        ----------
        is_reverse : bool
            Whether the pass moves towards earlier frames.
        is_start_frame_tracked : bool
            Whether an earlier pass already tracked the start frame, so this pass may skip it.
        start_frame_index : int
            First frame.
        max_frame_count : int | None
            Maximum number of frames after the start frame; ``None`` for the whole video.

        Yields
        ------
        ConceptFrame
            Instances with their tracks on each frame of the pass.
        """

    @abstractmethod
    def _track_next(self, frame: Image.Image, frame_index: int) -> ConceptFrame:
        """
        Add a frame to a stream and track the concepts on it.

        Parameters
        ----------
        frame : Image.Image
            RGB frame.
        frame_index : int
            Index of the frame in the stream.

        Returns
        -------
        ConceptFrame
            Instances with their tracks on the frame.
        """

    @abstractmethod
    def _reset(self) -> None:
        """
        Clear the model's tracks and tracking memory, keeping the prompt.
        """

    @abstractmethod
    def _remove_track(self, track_id: int) -> None:
        """
        Remove an existing track from the model state.

        Parameters
        ----------
        track_id : int
            Existing track identifier.
        """
