# Phase 4: Analytics engine

The analytics layer consumes confirmed tracks and source timestamps from
`FrameData`. Its results therefore remain reproducible when a video is replayed;
it does not use wall-clock time to infer dwell or bucket boundaries.

## Geometry and coordinates

Zones are polygons and counting lines are finite directed segments. Set
`normalized: true` on a zone or line to store coordinates in `[0, 1]`; each
coordinate is scaled by the current frame width or height when used. This makes
one zones file usable with different video resolutions. Pixel coordinates can
also be supplied. The tracked reference point defaults to bottom-center of the
bounding box (the estimated feet position).

As a starting point for 25-30 FPS footage, try 3 consecutive frames inside and
outside for debounce, a 1.0-second line cooldown, and
`dwell_min_seconds: 10`. Increase debounce for detector jitter, increase the
cooldown if people repeatedly oscillate at the line, and set dwell minimum from
the shortest meaningful queue visit in the footage.

The cross product of the directed line segment (`start` to `end`) and the
point's offset determines its side. A crossing from the negative side to the
positive side is **IN**; the reverse is **OUT**. A crossing is only counted if
the movement segment intersects the finite counting line.

## Events and output

`AnalyticsEngine.update(tracks, frame_data)` processes one source frame and
returns the events emitted for it. Sinks can keep events in memory or append
JSONL/CSV records to disk. Events include source time and frame ID. Use the
engine snapshot for current occupancy, line totals, and active track IDs; call
`finalize()` at end-of-stream to close remaining visits and flush sinks.

## Zones example

Copy `configs/zones.example.yaml` to `configs/zones.yaml`, then edit IDs,
labels, colors, and normalized points for the camera view. The interactive
`scripts/define_zones.py` tool can also create this file from a reference frame.

```shell
python scripts/define_zones.py --source data/raw/store.mp4
python scripts/define_zones.py --source data/raw/store.mp4 --zones configs/zones.yaml --preview
python scripts/run_analytics.py --source data/raw/store.mp4 --zones configs/zones.yaml --output outputs/analytics.mp4 --events-out outputs/events.jsonl --heatmap-out outputs/heatmap.png --report-out outputs/report
python scripts/replay_events.py --events outputs/events.jsonl
```

Running analytics requires the optional detection dependencies and model
weights described in the main README. Install the detection extra before
running against a video:

```shell
python -m pip install -e ".[detection]"
```

## Heatmaps and time series

The heatmap accumulator adds a precomputed Gaussian patch around each current
track position on a downscaled grid, so the update cost is proportional to the
number of tracks rather than the number of frame pixels. Cumulative heatmaps
preserve all activity; rolling heatmaps apply configured temporal decay.
Time-series rows use fixed-width buckets aligned to video time zero and export
as plain dictionaries or CSV without a pandas dependency.

## Reproducibility

The event replay tool recomputes event-derived totals and buckets from a saved
JSONL event stream. Detector and tracker output may depend on model/runtime
versions, but the analytics results for a fixed ordered track stream and
timestamps are deterministic.

## Synthetic overhead measurement

On the development machine, a 500-frame synthetic run with 20 tracks, 3 zones,
and 1 line measured a mean analytics update latency of **0.526 ms/frame**
(median 0.507 ms, p95 0.912 ms, p99 1.144 ms). This measures the analytics
engine only, not video decoding, model inference, annotation, or output writing;
performance depends on hardware and should be measured again for deployment.
