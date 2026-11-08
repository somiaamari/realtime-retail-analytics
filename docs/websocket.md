# WebSocket protocol

Connect to `/ws/streams/{stream_id}`. When `API_API_KEY` is configured, pass it
as `?api_key=...` (use TLS outside local development). Each client has a bounded
queue; slow clients may miss older events rather than back-pressuring capture.
The server sends `STATE` after connection, snapshots at 2 Hz, metrics at 1 Hz,
events as generated, and a heartbeat every 20 seconds.

All messages are JSON objects:

```json
{"type":"STATE","data":{"stream_id":"demo","state":"RUNNING"},"timestamp":1730000000.0}
{"type":"SNAPSHOT","data":{"timestamp_s":12.5,"frame_id":100,"occupancy":{"main":1},"line_counts":{},"active_tracks":[1],"unique_visitors":1},"timestamp":1730000000.5}
{"type":"METRICS","data":{"input_fps":8.0,"output_fps":8.0},"timestamp":1730000001.0}
{"type":"EVENT","data":{"event_type":"ZONE_ENTER","timestamp_s":12.5,"frame_id":100,"track_id":1,"zone_id":"main","line_id":null,"extra":{}}}
{"type":"HEARTBEAT","timestamp":1730000020.0}
```

Event timestamps use source-video seconds (`timestamp_s`); envelope timestamps
are Unix seconds. Track IDs are integers scoped to one pipeline run and are not
stable person identifiers.
