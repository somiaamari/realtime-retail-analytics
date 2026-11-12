import { http, HttpResponse } from "msw";
import type { Geometry, RetailEvent, Stream } from "../api/types";

const now = Date.now();
const baseStream: Stream = {
  stream_id: "demo-store", name: "Flagship · North entrance", source: "fake",
  autostart: true, status: "RUNNING",
  metrics: { input_fps: 29.8, processed_fps: 28.4, output_fps: 28.4, latency_p95_ms: 42, frames_dropped: 12 },
};
let streams = [baseStream, { ...baseStream, stream_id: "demo-floor-2", name: "Floor 02 · Home", status: "STOPPED" }];
let geometry: Geometry = {
  zones: [
    { id: "entrance", name: "Entrance", normalized: true, points: [[0.08, 0.2], [0.35, 0.18], [0.37, 0.68], [0.08, 0.7]] },
    { id: "main", name: "Main floor", normalized: true, points: [[0.42, 0.14], [0.91, 0.17], [0.88, 0.82], [0.4, 0.8]] },
  ],
  lines: [{ id: "entry-line", name: "Entry line", normalized: true, points: [[0.37, 0.22], [0.38, 0.72]] }],
};
const heatValues = Array.from({ length: 18 }, (_, y) =>
  Array.from({ length: 28 }, (_, x) => Math.max(0, Math.round(65 * Math.exp(-((x - 9) ** 2 + (y - 10) ** 2) / 40) + 35 * Math.exp(-((x - 21) ** 2 + (y - 6) ** 2) / 28)))),
);
const events: RetailEvent[] = Array.from({ length: 78 }, (_, index) => ({
  id: 1000 - index,
  stream_id: "demo-store",
  timestamp: new Date(now - index * 90_000).toISOString(),
  timestamp_s: 500 - index * 4,
  frame_id: 12000 - index * 80,
  track_id: 45 + index,
  event_type: index % 4 === 0 ? "LINE_CROSS" : index % 2 === 0 ? "ZONE_ENTER" : "ZONE_EXIT",
  zone_id: index % 3 === 0 ? "entrance" : "main",
  line_id: index % 4 === 0 ? "entry-line" : null,
  extra: {},
}));

export const handlers = [
  http.get("/api/v1/streams", () => HttpResponse.json(streams)),
  http.post("/api/v1/streams", async ({ request }) => {
    const body = await request.json() as { stream_id: string; name: string; source: string; autostart: boolean };
    const stream: Stream = { ...body, status: body.autostart ? "RUNNING" : "STOPPED" };
    streams = [...streams, stream];
    return HttpResponse.json(stream, { status: 201 });
  }),
  http.get("/api/v1/streams/:id", ({ params }) => {
    const stream = streams.find((item) => item.stream_id === params.id);
    return stream ? HttpResponse.json(stream) : HttpResponse.json({ detail: "Not found" }, { status: 404 });
  }),
  http.post("/api/v1/streams/:id/:action", ({ params }) => {
    streams = streams.map((stream) => stream.stream_id === params.id ? { ...stream, status: params.action === "stop" ? "STOPPED" : "RUNNING" } : stream);
    return HttpResponse.json(streams.find((stream) => stream.stream_id === params.id));
  }),
  http.delete("/api/v1/streams/:id", ({ params }) => {
    streams = streams.filter((stream) => stream.stream_id !== params.id);
    return new HttpResponse(null, { status: 204 });
  }),
  http.get("/api/v1/streams/:id/metrics", () => HttpResponse.json({ input_fps: 29.8, processed_fps: 28.4, output_fps: 28.4, latency_p95_ms: 42, frames_dropped: 12, stages: { capture: 6, detection: 24, tracking: 5, output: 7 } })),
  http.get("/api/v1/streams/:id/snapshot", () => HttpResponse.json({ snapshot: { occupancy: { entrance: 8, main: 26 }, active_tracks: Array.from({ length: 34 }, (_, index) => index) } })),
  http.get("/api/v1/streams/:id/analytics/summary", () => HttpResponse.json({ stream_id: "demo-store", event_count: 1284, event_counts: { ZONE_ENTER: 703, ZONE_EXIT: 581 }, unique_track_count: 986, zone_events: { entrance: 412, main: 872 }, average_dwell_seconds: 286 })),
  http.get("/api/v1/streams/:id/analytics/timeseries", () =>
    HttpResponse.json(Array.from({ length: 24 }, (_, i) => ({
      bucket_start: new Date(now - (23 - i) * 3_600_000).toISOString(),
      entries: Math.round(22 + 15 * Math.sin(i / 3) + (i % 5) * 4),
      exits: Math.round(18 + 11 * Math.sin(i / 3 - 0.8) + (i % 4) * 3),
      unique_visitors: 26 + (i * 13) % 46,
      mean_occupancy: Math.round(18 + 13 * Math.sin(i / 4)),
      peak_occupancy: Math.round(22 + 17 * Math.sin(i / 4)),
    }))),
  ),
  http.get("/api/v1/streams/:id/analytics/peak-hours", () =>
    HttpResponse.json(Array.from({ length: 24 }, (_, i) => ({ bucket_start: new Date(now - i * 3_600_000).toISOString(), peak_occupancy: 12 + (i * 7) % 48, entries: 20 + (i * 9) % 60 }))),
  ),
  http.get("/api/v1/streams/:id/analytics/zones", () => HttpResponse.json({ stream_id: "demo-store", zones: { entrance: { ZONE_ENTER: 412 }, main: { ZONE_ENTER: 872 } } })),
  http.get("/api/v1/streams/:id/analytics/dwell", () => HttpResponse.json({ zones: { entrance: { count: 318, average_seconds: 94, max_seconds: 342 }, main: { count: 668, average_seconds: 286, max_seconds: 910 } } })),
  http.get("/api/v1/streams/:id/analytics/heatmap", () => HttpResponse.json({ width: 28, height: 18, timestamp: new Date(now).toISOString(), values: heatValues })),
  http.get("/api/v1/streams/:id/events/export.csv", () => new HttpResponse("timestamp,event_type,zone_id\n2026-11-11T07:00:00Z,ZONE_ENTER,entrance\n", { headers: { "Content-Type": "text/csv" } })),
  http.get("/api/v1/streams/:id/events", ({ request }) => {
    const cursor = Number(new URL(request.url).searchParams.get("cursor") ?? 0);
    const items = events.filter((event) => cursor === 0 || event.id < cursor).slice(0, 50);
    return HttpResponse.json({ items, next_cursor: items.length === 50 ? items.at(-1)?.id : null });
  }),
  http.get("/api/v1/streams/:id/zones", () => HttpResponse.json(geometry)),
  http.put("/api/v1/streams/:id/zones", async ({ request }) => {
    geometry = await request.json() as Geometry;
    return HttpResponse.json(geometry);
  }),
];
