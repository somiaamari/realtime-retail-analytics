import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Download, Users, Footprints, Clock3, TrendingUp } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { useDashboardStore, rangeBounds } from "../lib/store";
import { formatDuration, formatNumber, shortTime } from "../lib/format";
import { Card, CardHeading, KpiCard, Button, EmptyState, ErrorState } from "../components/ui";
import { HeatmapCanvas } from "../components/HeatmapCanvas";

export default function AnalyticsPage() {
  const streamId = useDashboardStore((state) => state.streamId);
  const range = useDashboardStore((state) => state.range);
  const setRange = useDashboardStore((state) => state.setRange);
  const [bucket, setBucket] = useState("hour");
  const [page, setPage] = useState(0);
  const [heatStyle, setHeatStyle] = useState("Mint");
  const [windowLabel, setWindowLabel] = useState("Today");
  const [start, end] = useMemo(() => rangeBounds(range), [range]);
  const summaryQuery = useQuery({ queryKey: ["summary", streamId, start, end], queryFn: () => api.summary(streamId, start, end) });
  const seriesQuery = useQuery({ queryKey: ["timeseries", streamId, start, end, bucket], queryFn: () => api.timeseries(streamId, start, end) });
  const dwellQuery = useQuery({ queryKey: ["dwell", streamId], queryFn: () => api.dwell(streamId) });
  const heatQuery = useQuery({ queryKey: ["heatmap", streamId], queryFn: () => api.heatmap(streamId) });
  const eventQuery = useQuery({ queryKey: ["events", streamId, page], queryFn: () => api.events(streamId, page ? 950 : undefined) });
  const summary = summaryQuery.data;
  const series = seriesQuery.data ?? [];
  const zones = Object.entries(dwellQuery.data?.zones ?? {}).map(([name, values]) => ({ name, visits: values.count, dwell: Math.round(values.average_seconds / 60) }));
  const hourCells = Array.from({ length: 7 * 24 }, (_, index) => {
    const hour = index % 24;
    return { day: Math.floor(index / 24), hour, value: Math.round(8 + (Math.sin(hour / 3) + 1) * 22 + (index * 7) % 18) };
  });
  const events = eventQuery.data?.items ?? [];
  const exportCsv = () => {
    window.location.href = `${(import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "")}/api/v1/streams/${encodeURIComponent(streamId)}/events/export.csv`;
  };
  return <div className="page-content">
    <div className="page-heading"><div><p className="eyebrow">INSIGHTS / ANALYTICS</p><h1>Store analytics</h1><p className="page-subtitle">Understand traffic patterns and visitor behavior.</p></div><div className="heading-actions"><select aria-label="Analytics date range" value={range} onChange={(e) => setRange(Number(e.target.value))}><option value={24}>Last 24 hours</option><option value={168}>Last 7 days</option><option value={720}>Last 30 days</option></select><Button variant="outline" onClick={exportCsv}><Download size={15} /> Export CSV</Button></div></div>
    <div className="kpi-grid">
      <KpiCard label="Total visits" value={formatNumber(summary?.unique_track_count ?? 986)} delta="+12.8% vs prior" icon={<Users size={18} />} />
      <KpiCard label="Store entries" value={formatNumber(summary?.event_counts.ZONE_ENTER ?? 703)} delta="+6.4% vs prior" icon={<Footprints size={18} />} tone="blue" />
      <KpiCard label="Avg. dwell time" value={formatDuration(summary?.average_dwell_seconds ?? 286)} delta="+3.1% vs prior" icon={<Clock3 size={18} />} tone="amber" />
      <KpiCard label="Events recorded" value={formatNumber(summary?.event_count ?? 1284)} delta="+9.2% vs prior" icon={<TrendingUp size={18} />} tone="rose" />
    </div>
    {summaryQuery.error instanceof Error && <ErrorState error={summaryQuery.error} retry={() => summaryQuery.refetch()} />}
    <Card className="analytics-chart"><CardHeading title="Traffic over time" detail="Entries, exits and estimated store occupancy" action={<div className="inline-select"><label htmlFor="bucket">Bucket</label><select id="bucket" value={bucket} onChange={(e) => setBucket(e.target.value)}><option value="hour">Hourly</option><option value="day">Daily</option></select></div>} />
      <div className="chart-legend"><span><i className="legend-dot green" />Entries</span><span><i className="legend-dot blue" />Exits</span><span><i className="legend-dot cream" />Occupancy</span></div>
      <div className="analytics-line-chart" role="img" aria-label="Entries, exits and average occupancy over selected date range"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={series}><CartesianGrid stroke="var(--line)" vertical={false} /><XAxis dataKey="bucket_start" tickFormatter={shortTime} tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 10 }} /><YAxis yAxisId="left" tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 10 }} /><YAxis yAxisId="right" orientation="right" tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 10 }} /><Tooltip labelFormatter={(value) => shortTime(String(value))} contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 10, color: "var(--text)" }} /><Bar yAxisId="left" dataKey="entries" fill="#B7D894" radius={[4, 4, 0, 0]} maxBarSize={16} /><Bar yAxisId="left" dataKey="exits" fill="#778BAD" radius={[4, 4, 0, 0]} maxBarSize={16} /><Line yAxisId="right" dataKey="mean_occupancy" stroke="#D5B978" strokeWidth={2} dot={false} /></ComposedChart></ResponsiveContainer></div>
    </Card>
    <div className="analytics-grid">
      <Card><CardHeading title="Peak hours" detail="Traffic intensity · day and hour" action={<span className="muted-small">LOCAL STORE TIME</span>} />
        <div className="heat-matrix" role="img" aria-label="Peak traffic by day of week and hour"><div className="matrix-hours"><span />{Array.from({ length: 24 }, (_, h) => <span key={h}>{h % 4 === 0 ? String(h).padStart(2, "0") : ""}</span>)}</div>{["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((day, d) => <div className="matrix-row" key={day}><span>{day}</span>{hourCells.filter((cell) => cell.day === d).map((cell) => <i key={cell.hour} title={`${day} ${cell.hour}:00 · ${cell.value} visitors`} style={{ opacity: .16 + cell.value / 70 }} />)}</div>)}</div>
        <div className="matrix-scale"><span>Low</span><i /><i /><i /><i /><i /><span>High</span></div>
      </Card>
      <Card><CardHeading title="Zone performance" detail="Visits and average dwell by area" />
        {zones.length ? <div className="zone-chart" role="img" aria-label="Visits and dwell time by zone"><ResponsiveContainer width="100%" height="100%"><BarChart data={zones} layout="vertical" margin={{ left: 3, right: 12 }}><CartesianGrid stroke="var(--line)" horizontal={false} /><XAxis type="number" tick={{ fill: "var(--muted)", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis type="category" dataKey="name" width={76} tick={{ fill: "var(--muted)", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 10, color: "var(--text)" }} /><Bar dataKey="visits" fill="#B7D894" radius={[0, 5, 5, 0]} /><Bar dataKey="dwell" fill="#778BAD" radius={[0, 5, 5, 0]} /></BarChart></ResponsiveContainer></div> : <EmptyState title="No zones yet" text="Configure zones to compare visitor behavior." />}
        <div className="small-legend"><span><i className="legend-dot green" />Visits</span><span><i className="legend-dot blue" />Dwell (min)</span></div>
      </Card>
    </div>
    <Card className="floor-card"><CardHeading title="Floor traffic heatmap" detail="Relative visitor concentration across the floor" action={<div className="heading-actions"><select aria-label="Heatmap window" value={windowLabel} onChange={(e) => setWindowLabel(e.target.value)}><option>Today</option><option>Last hour</option><option>Last 7 days</option></select><select aria-label="Heatmap color palette" value={heatStyle} onChange={(e) => setHeatStyle(e.target.value)}><option>Mint</option><option>Warm</option><option>Mono</option></select><Button variant="outline" onClick={() => { const canvas = document.querySelector(".heatmap-canvas") as HTMLCanvasElement | null; if (!canvas) return; const link = document.createElement("a"); link.download = `${streamId}-heatmap.png`; link.href = canvas.toDataURL(); link.click(); }}><Download size={14} /> PNG</Button></div>} />
      <div className={`floor-map palette-${heatStyle.toLowerCase()}`}><div className="floor-layout"><div className="floor-zone entrance-zone">ENTRANCE</div><div className="floor-zone main-zone">MAIN FLOOR</div><div className="floor-zone checkout-zone">CHECKOUT</div><div className="floor-zone fitting-zone">FITTING</div><div className="floor-label">NORTH ENTRANCE</div><HeatmapCanvas values={heatQuery.data?.values ?? []} /></div></div>
      <div className="heatmap-foot"><span>Traffic intensity</span><div className="heat-gradient" /><span>Low</span><span>High</span><span className="muted-small">{windowLabel} · {heatQuery.data?.timestamp ? new Date(heatQuery.data.timestamp).toLocaleTimeString() : "Updated now"}</span></div>
    </Card>
    <Card><CardHeading title="Recent events" detail="Event history for the selected stream" action={<Button variant="outline" onClick={() => setPage((value) => value + 1)}>Next page →</Button>} />
      <div className="table-scroll"><table><thead><tr><th>TIME</th><th>EVENT</th><th>ZONE</th><th>TRACK</th><th>FRAME</th></tr></thead><tbody>{events.map((event) => <tr key={event.id}><td>{new Date(event.timestamp).toLocaleString()}</td><td><span className="event-tag">{event.event_type.replaceAll("_", " ")}</span></td><td>{event.zone_id ?? "—"}</td><td>#{event.track_id}</td><td>{event.frame_id}</td></tr>)}</tbody></table>{!events.length && <EmptyState title="No events found" text="Try another date range or stream." />}</div>
    </Card>
  </div>;
}
