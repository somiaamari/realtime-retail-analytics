import { useQuery } from "@tanstack/react-query";
import { Activity, Cpu, Database, Gauge, Radio, Server } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { useDashboardStore } from "../lib/store";
import { Badge, Card, CardHeading, KpiCard } from "../components/ui";
import { useStreamSocket } from "../realtime/useStreamSocket";

export default function SystemPage() {
  const streamId = useDashboardStore((state) => state.streamId);
  const live = useStreamSocket(streamId);
  const metricsQuery = useQuery({ queryKey: ["metrics", streamId], queryFn: () => api.metrics(streamId), refetchInterval: 5000 });
  const metrics = { ...metricsQuery.data, ...live.metrics };
  const stages = metrics.stages as Record<string, number> | undefined;
  const chartData = Object.entries(stages ?? { capture: 6, detection: 24, tracking: 5, output: 7 }).map(([name, duration]) => ({ name, duration }));
  const max = Math.max(1, ...chartData.map((item) => item.duration));
  const slowest = chartData.reduce((a, b) => a.duration > b.duration ? a : b);
  return <div className="page-content">
    <div className="page-heading"><div><p className="eyebrow">OBSERVABILITY / SYSTEM</p><h1>System health</h1><p className="page-subtitle">Live performance and pipeline health for the selected stream.</p></div><Badge tone={live.connected ? "good" : "warning"}>{live.connected ? "All systems operational" : "Connection reconnecting"}</Badge></div>
    <div className="kpi-grid">
      <KpiCard label="Input throughput" value={`${Number(metrics.input_fps ?? 0).toFixed(1)} fps`} icon={<Radio size={18} />} />
      <KpiCard label="Processed" value={`${Number(metrics.processed_fps ?? 0).toFixed(1)} fps`} icon={<Cpu size={18} />} tone="blue" />
      <KpiCard label="Output" value={`${Number(metrics.output_fps ?? 0).toFixed(1)} fps`} icon={<Gauge size={18} />} tone="amber" />
      <KpiCard label="Dropped frames" value={Number(metrics.frames_dropped ?? 0)} icon={<Activity size={18} />} tone="rose" />
    </div>
    <div className="system-grid">
      <Card><CardHeading title="Pipeline stage latency" detail="Average processing time per stage" action={<span className="muted-small">MILLISECONDS / FRAME</span>} />
        <div className="stage-bars">{chartData.map(({ name, duration }) => <div className={`stage-row ${name === slowest.name ? "bottleneck" : ""}`} key={name}><div><span>{name}</span><strong>{duration.toFixed(1)} ms</strong></div><div className="progress-track"><span style={{ width: `${duration / max * 100}%` }} /></div>{name === slowest.name && <span className="bottleneck-label">BOTTLENECK</span>}</div>)}</div>
      </Card>
      <Card><CardHeading title="Latency distribution" detail="Stage contribution · stacked view" />
        <div className="stage-chart" role="img" aria-label="Latency by pipeline stage bar chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={[{ stream: "Current", ...Object.fromEntries(chartData.map((stage) => [stage.name, stage.duration])) }]}><CartesianGrid stroke="var(--line)" vertical={false} /><XAxis dataKey="stream" tick={{ fill: "var(--muted)" }} axisLine={false} tickLine={false} /><YAxis tick={{ fill: "var(--muted)", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 10, color: "var(--text)" }} />{chartData.map((stage, index) => <Bar key={stage.name} dataKey={stage.name} stackId="a" fill={["#B7D894", "#778BAD", "#D5B978", "#94B6AA"][index % 4]} />)}</BarChart></ResponsiveContainer></div>
        <div className="small-legend">{chartData.map((stage, index) => <span key={stage.name}><i className="legend-dot" style={{ background: ["#B7D894", "#778BAD", "#D5B978", "#94B6AA"][index % 4] }} />{stage.name}</span>)}</div>
      </Card>
    </div>
    <div className="system-summary-grid"><Card><div className="summary-card"><Server size={18} /><div><span>Pipeline state</span><strong>{live.state}</strong></div><Badge tone={live.state === "RUNNING" ? "good" : "warning"}>Live</Badge></div></Card><Card><div className="summary-card"><Database size={18} /><div><span>Stream</span><strong>{streamId}</strong></div><span className="muted-small">API</span></div></Card><Card><div className="summary-card"><Activity size={18} /><div><span>p95 latency</span><strong>{Number(metrics.latency_p95_ms ?? 0)} ms</strong></div><span className="muted-small">RECENT</span></div></Card></div>
  </div>;
}
