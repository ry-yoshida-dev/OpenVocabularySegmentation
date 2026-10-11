from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from types import MappingProxyType

from image_container import BinaryImage
from open_vocabulary_detector import ImageSize
from PIL import Image

from ...image import ImageInput, RgbImage
from ...object_prompt import MaskLogits, ObjectPrompt
from ...options import PromptElement, PropagationDirection
from .tracked_frame import TrackedFrame


class TrackingSession(ABC):
    """
    Tracking state of one video: the prompted objects and the memory of the frames processed so far.

    Objects are identified by non-negative integers chosen by the caller. A prompt on a frame adds a new object or
    corrects an existing one on that frame, replacing what was prompted for it there before. Prompts are points
    and/or a box, or a binary mask alone; mask logits of image segmenters are rejected.

    An offline session (``ObjectTracker.start_session``) knows every frame: prompt objects on any frames with
    ``prompt_objects``, then ``propagate`` them. A streaming session (``ObjectTracker.start_stream``) receives frames
    one at a time through ``track_next``, which also takes the prompts of the new frame; it keeps only the memory the
    model reads, so its footprint stays bounded however long the stream runs.
    """

    def __init__(self, image_size: ImageSize | None, frame_count: int, is_streaming: bool) -> None:
        """
        Parameters
        ----------
        image_size : ImageSize | None
            Size of the frames; ``None`` for a stream before its first frame.
        frame_count : int
            Number of frames known to the session.
        is_streaming : bool
            Whether frames arrive through ``track_next``.
        """
        self._image_size: ImageSize | None = image_size
        self._frame_count: int = frame_count
        self._is_streaming: bool = is_streaming
        self._object_ids: list[int] = []

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

    @property
    def object_ids(self) -> tuple[int, ...]:
        """
        Identifiers of the prompted objects, in the order they were first prompted.

        Returns
        -------
        tuple[int, ...]
            Object identifiers.
        """
        return tuple(self._object_ids)

    def prompt_object(self, frame_index: int, object_id: int, prompt: ObjectPrompt) -> TrackedFrame:
        """
        Add or correct one object on one frame of an offline session.

        Parameters
        ----------
        frame_index : int
            Index of the frame the prompt is drawn on.
        object_id : int
            Non-negative identifier of the object.
        prompt : ObjectPrompt
            Points and/or box, or a binary mask, on the frame.

        Returns
        -------
        TrackedFrame
            Masks of every object on the frame after the prompt.
        """
        return self.prompt_objects(frame_index, {object_id: prompt})

    def prompt_objects(self, frame_index: int, prompts: Mapping[int, ObjectPrompt]) -> TrackedFrame:
        """
        Add or correct objects on one frame of an offline session.

        Parameters
        ----------
        frame_index : int
            Index of the frame the prompts are drawn on.
        prompts : Mapping[int, ObjectPrompt]
            Prompt of each object keyed by its non-negative identifier.

        Raises
        ------
        ValueError
            If the session is streaming, the frame index is out of range, there is no prompt, an identifier is
            negative or a prompt is invalid for tracking or the frame.

        Returns
        -------
        TrackedFrame
            Masks of every object on the frame after the prompts.
        """
        if self._is_streaming:
            raise ValueError("a streaming session takes prompts through track_next.")
        if not 0 <= frame_index < self._frame_count:
            raise ValueError(f"frame_index must be in [0, {self._frame_count}). got {frame_index}")
        if not prompts:
            raise ValueError("at least one prompt is needed.")
        self._validate_prompts(prompts, self._require_image_size())
        self._register(prompts)
        return self._prompt(frame_index, MappingProxyType(dict(prompts)))

    def propagate(
        self,
        direction: PropagationDirection = PropagationDirection.FORWARD,
        start_frame_index: int | None = None,
        max_frame_count: int | None = None,
    ) -> Iterator[TrackedFrame]:
        """
        Track the prompted objects through the video of an offline session.

        Parameters
        ----------
        direction : PropagationDirection, optional
            Forward, backward, or forward then backward from the start frame.
        start_frame_index : int | None, optional
            First frame of every pass; ``None`` starts at the earliest prompted frame.
        max_frame_count : int | None, optional
            Maximum number of frames per pass after the start frame; ``None`` runs to the end of the video.

        Raises
        ------
        ValueError
            If the session is streaming, no object is prompted, the start frame is out of range or
            ``max_frame_count`` is negative.

        Yields
        ------
        TrackedFrame
            Masks of every object on each frame, in processing order; a frame reached by both passes of a
            bidirectional propagation is yielded once.
        """
        if self._is_streaming:
            raise ValueError("a streaming session tracks frames through track_next.")
        if not self._object_ids:
            raise ValueError("prompt at least one object before propagating.")
        if start_frame_index is not None and not 0 <= start_frame_index < self._frame_count:
            raise ValueError(f"start_frame_index must be in [0, {self._frame_count}). got {start_frame_index}")
        if max_frame_count is not None and max_frame_count < 0:
            raise ValueError(f"max_frame_count must not be negative. got {max_frame_count}")
        yielded_frame_indices: set[int] = set()
        for is_reverse in direction.is_reverse_per_pass:
            for frame in self._propagate(is_reverse, start_frame_index, max_frame_count):
                if frame.frame_index in yielded_frame_indices:
                    continue
                yielded_frame_indices.add(frame.frame_index)
                yield frame

    def track_next(self, frame: ImageInput, prompts: Mapping[int, ObjectPrompt] | None = None) -> TrackedFrame:
        """
        Track the objects on the next frame of a stream, optionally adding or correcting objects on it.

        Frames received before the first object are stored without running the model and yield empty frames.

        Parameters
        ----------
        frame : ImageInput
            Next frame of the stream.
        prompts : Mapping[int, ObjectPrompt] | None, optional
            Prompts drawn on this frame, keyed by object identifier.

        Raises
        ------
        ValueError
            If the session is offline, the frame size differs from the previous frames, an identifier is negative or
            a prompt is invalid for tracking or the frame.

        Returns
        -------
        TrackedFrame
            Masks of every object on the frame.
        """
        if not self._is_streaming:
            raise ValueError("an offline session knows its frames; use prompt_objects and propagate.")
        rgb_frame: Image.Image = RgbImage.from_input(frame)
        frame_size: ImageSize = ImageSize.from_image(rgb_frame)
        if self._image_size is not None and frame_size != self._image_size:
            raise ValueError(
                f"stream frames must be {self._image_size.width}x{self._image_size.height}. "
                + f"got {frame_size.width}x{frame_size.height}"
            )
        frame_prompts: Mapping[int, ObjectPrompt] = MappingProxyType(dict(prompts or {}))
        self._validate_prompts(frame_prompts, frame_size)
        self._image_size = frame_size
        self._register(frame_prompts)
        tracked: TrackedFrame = self._track_next(rgb_frame, self._frame_count, frame_prompts)
        self._frame_count += 1
        return tracked

    def reset(self) -> None:
        """
        Forget every object and the tracking memory; an offline session keeps its frames.
        """
        self._object_ids.clear()
        self._reset()

    @abstractmethod
    def _prompt(self, frame_index: int, prompts: Mapping[int, ObjectPrompt]) -> TrackedFrame:
        """
        Add validated prompts on a frame of an offline session and segment the frame.

        Parameters
        ----------
        frame_index : int
            Index of the frame.
        prompts : Mapping[int, ObjectPrompt]
            Validated prompts keyed by object identifier.

        Returns
        -------
        TrackedFrame
            Masks of every object on the frame.
        """

    @abstractmethod
    def _propagate(
        self, is_reverse: bool, start_frame_index: int | None, max_frame_count: int | None
    ) -> Iterator[TrackedFrame]:
        """
        Run one propagation pass.

        Parameters
        ----------
        is_reverse : bool
            Whether the pass moves towards earlier frames.
        start_frame_index : int | None
            First frame; ``None`` for the earliest prompted frame.
        max_frame_count : int | None
            Maximum number of frames after the start frame; ``None`` for the whole video.

        Yields
        ------
        TrackedFrame
            Masks of every object on each frame of the pass.
        """

    @abstractmethod
    def _track_next(self, frame: Image.Image, frame_index: int, prompts: Mapping[int, ObjectPrompt]) -> TrackedFrame:
        """
        Add a frame to a stream with its validated prompts and segment it.

        Parameters
        ----------
        frame : Image.Image
            RGB frame.
        frame_index : int
            Index of the frame in the stream.
        prompts : Mapping[int, ObjectPrompt]
            Validated prompts of the frame keyed by object identifier; may be empty.

        Returns
        -------
        TrackedFrame
            Masks of every object on the frame.
        """

    @abstractmethod
    def _reset(self) -> None:
        """
        Clear the model's tracking state.
        """

    def _require_image_size(self) -> ImageSize:
        if self._image_size is None:
            raise ValueError("the session has no frame yet.")
        return self._image_size

    def _register(self, prompts: Mapping[int, ObjectPrompt]) -> None:
        for object_id in prompts:
            if object_id not in self._object_ids:
                self._object_ids.append(object_id)

    @staticmethod
    def _validate_prompts(prompts: Mapping[int, ObjectPrompt], image_size: ImageSize) -> None:
        for object_id, prompt in prompts.items():
            if object_id < 0:
                raise ValueError(f"object identifiers must not be negative. got {object_id}")
            match prompt.mask:
                case None:
                    pass
                case MaskLogits():
                    raise ValueError("video trackers take binary masks; mask logits only refine image segmenters.")
                case BinaryImage():
                    if prompt.elements != frozenset({PromptElement.MASK}):
                        raise ValueError("a video mask prompt cannot be combined with points or a box.")
            prompt.validate_for(image_size)
