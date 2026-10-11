from dataclasses import dataclass
from typing import cast

from transformers.models.sam3_video.modeling_sam3_video import Sam3VideoInferenceSession


@dataclass(frozen=True)
class FusionStatePruner:
    """
    Frees the detection-tracking fusion records of a streamed SAM 3 video session that are never read again.

    Per-frame tracker scores and suppressed tracks are only read on their own frame, and the unmatched and overlapping
    frame lists only feed the hot start removal, which ``transformers`` skips in streams. The per-track records of a
    removed track are deleted with it.
    """

    def prune(self, inference_session: Sam3VideoInferenceSession, frame_index: int) -> None:
        """
        Free the per-frame records older than ``frame_index``.

        Parameters
        ----------
        inference_session : Sam3VideoInferenceSession
            Streamed session.
        frame_index : int
            Index of the frame just processed.
        """
        tracker_scores_by_frame: dict[int, object] = cast(
            dict[int, object], inference_session.obj_id_to_tracker_score_frame_wise
        )
        suppressed_tracks_by_frame: dict[int, object] = cast(dict[int, object], inference_session.suppressed_obj_ids)
        for records_by_frame in (tracker_scores_by_frame, suppressed_tracks_by_frame):
            for stale_index in [index for index in records_by_frame if index < frame_index]:
                del records_by_frame[stale_index]
        inference_session.unmatched_frame_inds.clear()
        inference_session.overlap_pair_to_frame_inds.clear()

    def forget_track(self, inference_session: Sam3VideoInferenceSession, track_id: int) -> None:
        """
        Delete the per-track records of a track removed from a stream.

        Parameters
        ----------
        inference_session : Sam3VideoInferenceSession
            Streamed session.
        track_id : int
            Removed track.
        """
        inference_session.obj_id_to_score.pop(track_id, None)
        inference_session.obj_id_to_last_occluded.pop(track_id, None)
        inference_session.obj_first_frame_idx.pop(track_id, None)
        inference_session.trk_keep_alive.pop(track_id, None)
