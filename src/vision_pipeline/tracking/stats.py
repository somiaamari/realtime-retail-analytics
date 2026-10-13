"""Cumulative identity and track-lifetime statistics."""

from collections.abc import Sequence

from vision_pipeline.types import Track, TrackState


class TrackingStats:
    """Accumulate confirmed unique IDs, activity, lifetime, and recoveries."""

    def __init__(self) -> None:
        """Initialize empty statistics for one tracker run."""
        self.reset()

    def reset(self) -> None:
        """Clear accumulated values and per-ID lifetime bookkeeping."""
        self._confirmed_ids: set[int] = set()
        self._max_age_by_id: dict[int, int] = {}
        self.total_unique_ids = 0
        self.active_count = 0
        self.average_track_lifetime = 0.0
        self.reidentifications = 0
        self.per_frame_active_counts: list[int] = []

    def update(self, tracks: Sequence[Track]) -> None:
        """Update cumulative metrics from the tracker state after one frame.

        Args:
            tracks: Current tracklets, including lost and newly removed tracks.
        """
        for track in tracks:
            if track.reidentified:
                self.reidentifications += 1
            if track.state in (TrackState.TRACKED, TrackState.LOST):
                self._confirmed_ids.add(track.track_id)
            if track.track_id in self._confirmed_ids:
                self._max_age_by_id[track.track_id] = max(
                    track.age,
                    self._max_age_by_id.get(track.track_id, 0),
                )
        self.total_unique_ids = len(self._confirmed_ids)
        self.active_count = sum(
            track.state is TrackState.TRACKED and track.time_since_update == 0
            for track in tracks
        )
        self.per_frame_active_counts.append(self.active_count)
        lifetimes = [self._max_age_by_id[track_id] for track_id in self._confirmed_ids]
        self.average_track_lifetime = (
            sum(lifetimes) / len(lifetimes) if lifetimes else 0.0
        )
