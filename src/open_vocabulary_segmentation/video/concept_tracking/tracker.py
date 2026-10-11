from abc import ABC, abstractmethod
from collections.abc import Sequence

from open_vocabulary_detector import Prompt, PromptKind
from PIL import Image

from ...image import ImageInput, RgbImage
from .session import ConceptTrackingSession
from .settings import ConceptTrackerSettings


class ConceptTracker(ABC):
    """
    Common base of video concept trackers: detects every instance of the text queries of a prompt and tracks each
    with a stable identifier (SAM 3 video).

    Attributes
    ----------
    settings : ConceptTrackerSettings
        Weights, thresholds, device and precision.
    """

    def __init__(self, settings: ConceptTrackerSettings) -> None:
        """
        Parameters
        ----------
        settings : ConceptTrackerSettings
            Weights, thresholds, device and precision.
        """
        self.settings: ConceptTrackerSettings = settings

    @property
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        """
        Query kinds the tracker accepts.

        Returns
        -------
        frozenset[PromptKind]
            ``TEXT`` only.
        """
        return frozenset({PromptKind.TEXT})

    def start_session(self, frames: Sequence[ImageInput], prompt: Prompt) -> ConceptTrackingSession:
        """
        Start tracking the concepts of a prompt in a whole video.

        Parameters
        ----------
        frames : Sequence[ImageInput]
            Frames of the video in order, all of one size.
        prompt : Prompt
            Classes and their text queries.

        Raises
        ------
        ValueError
            If there is no frame, the frames differ in size or the prompt has unsupported queries.

        Returns
        -------
        ConceptTrackingSession
            Session over the video.
        """
        self._validate_prompt(prompt)
        if not frames:
            raise ValueError("a video needs at least one frame.")
        rgb_frames: list[Image.Image] = [RgbImage.from_input(frame) for frame in frames]
        sizes: set[tuple[int, int]] = {frame.size for frame in rgb_frames}
        if len(sizes) != 1:
            raise ValueError(f"every frame must have the same size. got {sorted(sizes)}")
        return self._start_offline(rgb_frames, prompt)

    def start_stream(self, prompt: Prompt) -> ConceptTrackingSession:
        """
        Start tracking the concepts of a prompt in frames that arrive one by one.

        Parameters
        ----------
        prompt : Prompt
            Classes and their text queries.

        Raises
        ------
        ValueError
            If the prompt has unsupported queries.

        Returns
        -------
        ConceptTrackingSession
            Streaming session; feed it with ``ConceptTrackingSession.track_next``.
        """
        self._validate_prompt(prompt)
        return self._start_streaming(prompt)

    def _validate_prompt(self, prompt: Prompt) -> None:
        unsupported_kinds: frozenset[PromptKind] = prompt.kinds - self.supported_prompt_kinds
        if unsupported_kinds:
            raise ValueError(
                f"{type(self).__name__} does not support {sorted(kind.value for kind in unsupported_kinds)} queries."
            )

    @abstractmethod
    def _start_offline(self, frames: Sequence[Image.Image], prompt: Prompt) -> ConceptTrackingSession:
        """
        Preprocess a whole video and open a session over it.

        Parameters
        ----------
        frames : Sequence[Image.Image]
            RGB frames of one size.
        prompt : Prompt
            Validated prompt.

        Returns
        -------
        ConceptTrackingSession
            Offline session.
        """

    @abstractmethod
    def _start_streaming(self, prompt: Prompt) -> ConceptTrackingSession:
        """
        Open a streaming session.

        Parameters
        ----------
        prompt : Prompt
            Validated prompt.

        Returns
        -------
        ConceptTrackingSession
            Streaming session.
        """
