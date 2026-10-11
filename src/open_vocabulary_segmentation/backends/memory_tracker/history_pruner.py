from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass(frozen=True)
class HistoryPruner:
    """
    Frees the parts of a streamed ``transformers`` video session the SAM 2-style memory never reads again.

    The memory reads the conditioning (prompted) frames, the memories of the last ``num_maskmem`` frames and the object
    pointers of the last ``max_object_pointers_in_encoder`` frames. Older non-conditioning outputs are deleted, and
    only the first and the latest ``conditioning_frame_limit - 1`` conditioning outputs of each object are kept, so
    a stream prompted again and again stays bounded. Point and mask inputs are only read on their own frame and are
    deleted afterwards. Processed frame pixels are only read for the current frame, but the frame count of the
    session bounds which object pointers are read, so older pixels are replaced by an empty tensor instead of being
    deleted.

    Attributes
    ----------
    history_length : int
        Number of past frames whose outputs are kept.
    conditioning_frame_limit : int
        Maximum number of conditioning frames kept per object.

    Raises
    ------
    ValueError
        If ``history_length`` is not positive or ``conditioning_frame_limit`` is below 2.
    """

    CONDITIONING_OUTPUTS_KEY: ClassVar[str] = "cond_frame_outputs"
    NON_CONDITIONING_OUTPUTS_KEY: ClassVar[str] = "non_cond_frame_outputs"

    history_length: int
    conditioning_frame_limit: int

    def __post_init__(self) -> None:
        if self.history_length <= 0:
            raise ValueError(f"history_length must be positive. got {self.history_length}")
        if self.conditioning_frame_limit < 2:
            raise ValueError(f"conditioning_frame_limit must be at least 2. got {self.conditioning_frame_limit}")

    def prune(
        self,
        frame_index: int,
        processed_frames: dict[int, torch.Tensor] | None,
        output_dicts: dict[int, dict[str, dict[int, object]]],
        tracked_frames_per_object: dict[int, dict[int, object]],
        inputs_per_object: tuple[dict[int, dict[int, object]], ...],
    ) -> None:
        """
        Free the history older than what the frame after ``frame_index`` reads.

        Parameters
        ----------
        frame_index : int
            Index of the frame just processed.
        processed_frames : dict[int, torch.Tensor] | None
            Preprocessed frames of the session by index.
        output_dicts : dict[int, dict[str, dict[int, object]]]
            Conditioning and non-conditioning outputs of every object by frame index.
        tracked_frames_per_object : dict[int, dict[int, object]]
            Frames each object was tracked on.
        inputs_per_object : tuple[dict[int, dict[int, object]], ...]
            Point and mask inputs of every object by frame index.
        """
        if processed_frames is not None:
            for stale_index, pixels in processed_frames.items():
                if stale_index < frame_index and pixels.numel():
                    processed_frames[stale_index] = pixels.new_empty(0)
        oldest_kept_index: int = frame_index - self.history_length
        for outputs in output_dicts.values():
            self._delete_before(outputs[self.NON_CONDITIONING_OUTPUTS_KEY], oldest_kept_index)
            self._limit_conditioning(outputs[self.CONDITIONING_OUTPUTS_KEY])
        for tracked_frames in tracked_frames_per_object.values():
            self._delete_before(tracked_frames, oldest_kept_index)
        for inputs in inputs_per_object:
            for frame_inputs in inputs.values():
                self._delete_before(frame_inputs, frame_index)

    def _limit_conditioning(self, conditioning_outputs: dict[int, object]) -> None:
        if len(conditioning_outputs) <= self.conditioning_frame_limit:
            return
        frame_indices: list[int] = sorted(conditioning_outputs)
        latest_count: int = self.conditioning_frame_limit - 1
        kept_indices: set[int] = {frame_indices[0], *frame_indices[-latest_count:]}
        for stale_index in frame_indices:
            if stale_index not in kept_indices:
                del conditioning_outputs[stale_index]

    @staticmethod
    def _delete_before(entries: dict[int, object], oldest_kept_index: int) -> None:
        for stale_index in [index for index in entries if index < oldest_kept_index]:
            del entries[stale_index]
