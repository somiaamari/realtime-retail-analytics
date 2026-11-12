import type {
  AnalyticsSummary,
  Geometry,
  Heatmap,
  Metrics,
  RetailEvent,
  Snapshot,
  Stream,
  TrafficBucket,
} from "./types";
import type { components } from "./schema";

const apiBase = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const prefix = "/api/v1";

export function getApiKey(): string | undefined {
  return typeof sessionStorage === "undefined" ? undefined : sessionStorage.getItem("rv-api-key") ?? undefined;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const apiKey = getApiKey();
  if (apiKey) headers.set("X-API-Key", apiKey);
  if (init?.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(`${apiBase}${prefix}${path}`, { ...init, headers });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`API ${response.status}: ${detail || response.statusText}`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  streams: () => request<Stream[]>("/streams"),
  createStream: (stream: Pick<components["schemas"]["StreamCreate"], "stream_id" | "name" | "source" | "autostart">) =>
    request<Stream>("/streams", { method: "POST", body: JSON.stringify(stream) }),
  streamAction: (id: string, action: "start" | "stop" | "restart") =>
    request<Stream>(`/streams/${encodeURIComponent(id)}/${action}`, { method: "POST" }),
  deleteStream: (id: string) =>
    request<void>(`/streams/${encodeURIComponent(id)}`, { method: "DELETE" }),
  metrics: (id: string) => request<Metrics>(`/streams/${encodeURIComponent(id)}/metrics`),
  snapshot: (id: string) => request<{ snapshot: Snapshot | null }>(`/streams/${encodeURIComponent(id)}/snapshot`),
  summary: (id: string, start: string, end: string) =>
    request<AnalyticsSummary>(`/streams/${encodeURIComponent(id)}/analytics/summary?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`),
  timeseries: (id: string, start: string, end: string) =>
    request<TrafficBucket[]>(`/streams/${encodeURIComponent(id)}/analytics/timeseries?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`),
  peakHours: (id: string) => request<Array<{ bucket_start: string; peak_occupancy: number; entries: number }>>(`/streams/${encodeURIComponent(id)}/analytics/peak-hours?limit=24`),
  dwell: (id: string) => request<{ zones: Record<string, { count: number; average_seconds: number; max_seconds: number }> }>(`/streams/${encodeURIComponent(id)}/analytics/dwell`),
  heatmap: (id: string) => request<Heatmap>(`/streams/${encodeURIComponent(id)}/analytics/heatmap`),
  events: (id: string, cursor?: number) =>
    request<{ items: RetailEvent[]; next_cursor: number | null }>(`/streams/${encodeURIComponent(id)}/events?limit=50${cursor ? `&cursor=${cursor}` : ""}`),
  geometry: (id: string) => request<Geometry>(`/streams/${encodeURIComponent(id)}/zones`),
  saveGeometry: (id: string, geometry: Geometry) =>
    request<Geometry>(`/streams/${encodeURIComponent(id)}/zones`, { method: "PUT", body: JSON.stringify(geometry) }),
  videoUrl: (id: string) => `${apiBase}${prefix}/streams/${encodeURIComponent(id)}/video`,
  snapshotUrl: (id: string) => `${apiBase}${prefix}/streams/${encodeURIComponent(id)}/frame.jpg`,
};

export { apiBase };
