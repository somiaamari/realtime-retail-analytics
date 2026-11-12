import type { SocketMessage } from "./protocol";

export class MockSocket {
  private timer: number | undefined;
  private count = 18;
  private tick = 0;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor() {
    window.setTimeout(() => {
      this.onopen?.();
      this.send({ type: "STATE", data: { state: "RUNNING", stream_id: "demo-store" } });
    }, 0);
    this.timer = window.setInterval(() => {
      this.tick += 1;
      this.count = Math.max(4, this.count + (this.tick % 3) - 1);
      const now = Date.now() / 1000;
      this.send({ type: "SNAPSHOT", timestamp: now, data: { occupancy: { entrance: Math.round(this.count * 0.3), main: Math.round(this.count * 0.7) }, active_tracks: Array.from({ length: this.count }, (_, i) => i + 1) } });
      this.send({ type: "METRICS", timestamp: now, data: { input_fps: 29.8, processed_fps: 28.4, output_fps: 28.4, latency_p95_ms: 42, frames_dropped: 12 } });
      if (this.tick % 7 === 0) {
        this.send({ type: "EVENT", timestamp: now, data: { event_type: this.tick % 14 === 0 ? "ZONE_EXIT" : "ZONE_ENTER", timestamp_s: this.tick, frame_id: this.tick * 30, track_id: 45 + this.tick, zone_id: this.tick % 14 === 0 ? "main" : "entrance", line_id: null, extra: {} } });
      }
      if (this.tick % 20 === 0) this.send({ type: "HEARTBEAT", timestamp: now });
    }, 1000);
  }

  private send(message: SocketMessage): void {
    this.onmessage?.({ data: JSON.stringify(message) } as MessageEvent<string>);
  }

  close(): void {
    if (this.timer !== undefined) window.clearInterval(this.timer);
    this.onclose?.();
  }
}
