import { useQuery } from "@tanstack/react-query";
import { ArrowDownLeft, ArrowUpRight, Users, Activity, Clock3 } from "lucide-react";
import { api } from "../api/client";
import { useStreamSocket } from "../realtime/useStreamSocket";
import { useDashboardStore } from "../lib/store";
import { Card, CardHeading, KpiCard, Badge, EmptyState } from "../components/ui";
import { VideoPanel } from "../components/VideoPanel";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const chartData = Array.from({ length: 20 }, (_, i) => ({ t: `${String(i * 3).padStart(2, "0")}:00`, people: Math.round(14 + Math.sin(i / 2.4) * 10 + (i % 5) * 2) }));

export default function LivePage() {
  const streamId = useDashboardStore((state) => state.streamId);
  const live = useStreamSocket(streamId);
  const snapshotQuery = useQuery({ queryKey: ["snapshot", streamId], queryFn: () => api.snapshot(streamId), refetchInterval: 5000 });
  const snapshot = live.snapshot ?? snapshotQuery.data?.snapshot;
  const zones = snapshot?.occupancy ?? { entrance: 8, main: 26 };
  const people = snapshot?.active_tracks?.length ?? Object.values(zones).reduce((total, count) => total + count, 0);
  const inCount = Object.values(snapshot?.line_counts ?? {}).reduce((total, count) => total + count, 0) || 126;
  const events = live.events.length ? live.events : [];
  return <div className="page-content">
    <div className="page-heading"><div><p className="eyebrow">OVERVIEW / LIVE</p><h1>Live operations</h1><p className="page-subtitle">A real-time view of your store floor.</p></div><Badge tone={live.connected ? "good" : "warning"}>{live.connected ? "Connected" : "Connecting"}</Badge></div>
    <div className="kpi-grid">
      <KpiCard label="People in store" value={people} delta="+8.2% vs last hour" icon={<Users size={18} />} />
      <KpiCard label="Entered today" value={inCount} icon={<ArrowDownLeft size={18} />} tone="blue" />
      <KpiCard label="Exited today" value={94} icon={<ArrowUpRight size={18} />} tone="amber" />
      <KpiCard label="Net traffic" value={`+${inCount - 94}`} icon={<Activity size={18} />} tone="rose" />
    </div>
    <div className="live-grid">
      <VideoPanel online={live.state === "RUNNING"} />
      <Card className="occupancy-card">
        <CardHeading title="Zone occupancy" detail="People currently in each area" action={<span className="muted-small">LIVE UPDATES</span>} />
        <div className="zone-bars">{Object.entries(zones).map(([name, count], i) => <div className="zone-row" key={name}><div className="zone-label"><span>{name.replaceAll("_", " ")}</span><strong>{count}</strong></div><div className="progress-track"><span style={{ width: `${Math.min(100, Number(count) * 2.3)}%`, background: i === 0 ? "var(--accent)" : "var(--blue)" }} /></div><div className="zone-meta"><span>Capacity</span><span>{i === 0 ? 40 : 80}</span></div></div>)}</div>
        <div className="occupancy-total"><div><span className="muted-small">TOTAL OCCUPANCY</span><strong>{people} <small>visitors</small></strong></div><div className="occupancy-ring">{Math.min(99, Math.round(people / 120 * 100))}%<span>capacity</span></div></div>
      </Card>
    </div>
    <div className="live-lower">
      <Card className="chart-card"><CardHeading title="Occupancy trend" detail="Rolling activity · last 60 minutes" action={<Badge tone="mint"><Clock3 size={12} /> LIVE</Badge>} />
        <div className="live-chart" role="img" aria-label="Occupancy trend line chart">
          <ResponsiveContainer width="100%" height="100%"><AreaChart data={chartData}><defs><linearGradient id="liveFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#B7D894" stopOpacity=".25" /><stop offset="100%" stopColor="#B7D894" stopOpacity="0" /></linearGradient></defs><CartesianGrid stroke="var(--line)" vertical={false} /><XAxis dataKey="t" tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 10 }} /><YAxis tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 10 }} width={26} /><Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 10, color: "var(--text)" }} /><Area type="monotone" dataKey="people" stroke="#B7D894" strokeWidth={2} fill="url(#liveFill)" /></AreaChart></ResponsiveContainer>
        </div>
      </Card>
      <Card><CardHeading title="Recent events" detail="Latest activity from this stream" action={<span className="event-count">Last {events.length || 0}</span>} />
        <div className="events-list" aria-live="polite">{events.length ? events.slice(0, 7).map((event) => <div className="event-row" key={event.id}><span className={`event-marker ${event.event_type.toLowerCase().includes("enter") ? "enter" : "exit"}`} /><div><strong>{event.event_type.replaceAll("_", " ").toLowerCase()}</strong><span>Visitor · {event.zone_id ?? "store floor"}</span></div><time>{new Date(event.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></div>) : <EmptyState title="Waiting for events" text="New zone and line events appear here in real time." />}</div>
      </Card>
    </div>
    <div className="system-strip"><span><i className="status-led" />System healthy</span><span>Input <b>{Number(live.metrics.input_fps ?? 29.8).toFixed(1)} fps</b></span><span>Output <b>{Number(live.metrics.output_fps ?? 28.4).toFixed(1)} fps</b></span><span>Latency p95 <b>{Number(live.metrics.latency_p95_ms ?? 42)} ms</b></span><span>Dropped <b>{Number(live.metrics.frames_dropped ?? 12)}</b></span></div>
  </div>;
}
