export interface Stream {
  stream_id: string;
  name: string;
  source: string;
  autostart: boolean;
  status: string;
  health?: Record<string, unknown> | null;
  metrics?: Metrics | null;
}

export interface Metrics {
  input_fps?: number;
  processed_fps?: number;
  output_fps?: number;
  latency_p50_ms?: number;
  latency_p95_ms?: number;
  frames_dropped?: number;
  stages?: Record<string, number>;
  [key: string]: unknown;
}

export interface Snapshot {
  timestamp_s?: number;
  frame_id?: number;
  occupancy?: Record<string, number>;
  line_counts?: Record<string, number>;
  active_tracks?: number[];
  unique_visitors?: number;
}

export interface TrafficBucket {
  bucket_start: string;
  entries: number;
  exits: number;
  unique_visitors: number;
  mean_occupancy: number;
  peak_occupancy: number;
  zones?: Record<string, unknown>;
}

export interface AnalyticsSummary {
  event_count: number;
  event_counts: Record<string, number>;
  unique_track_count: number;
  zone_events: Record<string, number>;
  average_dwell_seconds: number;
}

export interface RetailEvent {
  id: number;
  stream_id: string;
  timestamp: string;
  timestamp_s: number;
  frame_id: number;
  track_id: number;
  event_type: string;
  zone_id: string | null;
  line_id: string | null;
  extra: Record<string, unknown>;
}

export interface Geometry {
  zones: Array<{ id: string; name: string; points: number[][]; normalized?: boolean; [key: string]: unknown }>;
  lines: Array<{ id: string; name: string; points: number[][]; normalized?: boolean; [key: string]: unknown }>;
}

export interface Heatmap {
  width?: number;
  height?: number;
  values: number[][];
  timestamp?: string | null;
}

export interface ApiEnvelope<T> {
  type: "STATE" | "SNAPSHOT" | "METRICS" | "EVENT" | "HEARTBEAT";
  data?: T;
  timestamp?: number;
}
