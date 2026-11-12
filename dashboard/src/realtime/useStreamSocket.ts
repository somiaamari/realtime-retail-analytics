import { useEffect, useRef, useState } from "react";
import type { Metrics, RetailEvent, Snapshot } from "../api/types";
import { MockSocket } from "./mockSocket";
import { parseSocketMessage } from "./protocol";
import { getApiKey } from "../api/client";

export interface LiveState {
  state: string;
  snapshot: Snapshot | null;
  metrics: Metrics;
  events: RetailEvent[];
  connected: boolean;
}

const initial: LiveState = { state: "CONNECTING", snapshot: null, metrics: {}, events: [], connected: false };
const mockMode = import.meta.env.VITE_USE_MOCKS === "true";

export function useStreamSocket(streamId: string): LiveState {
  const [live, setLive] = useState(initial);
  const pending = useRef<Partial<LiveState>>({});
  const flush = useRef<number | undefined>(undefined);

  useEffect(() => {
    let active = true;
    let socket: WebSocket | MockSocket | undefined;
    let retry = 500;
    let reconnect: number | undefined;

    const schedule = () => {
      if (!active || document.hidden) return;
      reconnect = window.setTimeout(connect, retry);
      retry = Math.min(retry * 2, 15_000);
    };
    const connect = () => {
      if (!active || document.hidden) return;
      const base = (import.meta.env.VITE_API_BASE_URL ?? window.location.origin).replace(/^http/, "ws").replace(/\/$/, "");
      const key = getApiKey();
      const query = key ? `?api_key=${encodeURIComponent(key)}` : "";
      socket = mockMode ? new MockSocket() : new WebSocket(`${base}/ws/streams/${encodeURIComponent(streamId)}${query}`);
      socket.onopen = () => { retry = 500; queue({ connected: true, state: "RUNNING" }); };
      socket.onclose = () => { queue({ connected: false, state: "RECONNECTING" }); schedule(); };
      socket.onmessage = (event: MessageEvent<string>) => {
        const message = parseSocketMessage(event.data);
        if (!message) return;
        if (message.type === "STATE") queue({ state: String((message.data as { state?: unknown } | undefined)?.state ?? "RUNNING") });
        if (message.type === "SNAPSHOT") queue({ snapshot: (message.data as Snapshot | null) ?? null });
        if (message.type === "METRICS") queue({ metrics: (message.data as Metrics) ?? {} });
        if (message.type === "EVENT" && message.data) {
          const eventData = message.data as Partial<RetailEvent>;
          const event: RetailEvent = {
            id: Date.now(), stream_id: streamId, timestamp: new Date().toISOString(),
            timestamp_s: Number(eventData.timestamp_s ?? 0), frame_id: Number(eventData.frame_id ?? 0),
            track_id: Number(eventData.track_id ?? 0), event_type: String(eventData.event_type ?? "EVENT"),
            zone_id: eventData.zone_id ?? null, line_id: eventData.line_id ?? null,
            extra: eventData.extra ?? {},
          };
          setLive((old) => ({ ...old, events: [event, ...old.events].slice(0, 100) }));
        }
      };
    };
    function queue(next: Partial<LiveState>) {
      pending.current = { ...pending.current, ...next };
      if (flush.current === undefined) flush.current = window.setTimeout(() => {
        if (active) setLive((old) => ({ ...old, ...pending.current }));
        pending.current = {};
        flush.current = undefined;
      }, 120);
    }
    const onVisibility = () => {
      if (document.hidden) {
        if (reconnect !== undefined) window.clearTimeout(reconnect);
        socket?.close();
      } else connect();
    };
    connect();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      active = false;
      document.removeEventListener("visibilitychange", onVisibility);
      if (reconnect !== undefined) window.clearTimeout(reconnect);
      if (flush.current !== undefined) window.clearTimeout(flush.current);
      socket?.close();
    };
  }, [streamId]);

  return live;
}
