# Phase 3: Multi-object tracking

`vision_pipeline.tracking.ByteTracker` consumes the detector's `Detection`
objects and returns confirmed `Track` objects. It implements ByteTrack's
two-stage association: high-confidence detections are matched first, then
unmatched active tracks are matched with low-confidence detections. Low-score
detections can preserve an identity, but never create a new track.

```python
from vision_pipeline.tracking import ByteTracker

tracker = ByteTracker(config.tracker)
tracks = tracker.update(detections, frame_id=frame.frame_id)
unique_people_seen = tracker.stats.total_unique_ids
```

The tracker uses a pure-NumPy constant-velocity Kalman filter and SciPy's
`linear_sum_assignment`. Track IDs increase monotonically during a run;
calling `reset()` starts a new run and restarts IDs. Only tracks that have
reached `min_hits` and matched a detection in the current frame are returned.
Track `history` stores a bounded deque of bottom-center points for drawing
trails with `draw_tracks`.

Existing `track_thresh`, `match_thresh`, and `track_buffer` configuration
remains valid. `high_thresh` defaults to `track_thresh`; `low_thresh` defaults
to 0.1 but is reduced when needed to support older custom `track_thresh`
values. Other settings, including `min_hits`, `min_box_area`, `history_len`,
and `fuse_score`, are optional fields on `TrackerConfig`.

To compare the adapter with Supervision's reference implementation, install
the optional extra with `pip install -e ".[tracking-ref]"`. The reference
adapter imports Supervision lazily; core tracking and synthetic unit tests do
not require it, PyTorch, model weights, or network access.

Run the offline tests with `pytest tests/test_tracking.py tests/test_config.py`.
The optional reference comparison is in `tests/test_tracking_reference.py` and
is skipped when the extra is unavailable.
