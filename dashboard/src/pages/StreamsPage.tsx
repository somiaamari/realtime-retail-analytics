import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Camera, CirclePlus, Play, RotateCw, Square, Trash2 } from "lucide-react";
import { api } from "../api/client";
import type { Stream } from "../api/types";
import { useDashboardStore } from "../lib/store";
import { Badge, Button, Card, CardHeading, EmptyState, ErrorState, Skeleton } from "../components/ui";

export default function StreamsPage() {
  const client = useQueryClient();
  const selectStream = useDashboardStore((state) => state.setStreamId);
  const query = useQuery({ queryKey: ["streams"], queryFn: api.streams });
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState({ stream_id: "", name: "", source: "fake" });
  const lifecycle = useMutation({ mutationFn: ({ id, action }: { id: string; action: "start" | "stop" | "restart" }) => api.streamAction(id, action), onSuccess: () => client.invalidateQueries({ queryKey: ["streams"] }) });
  const create = useMutation({ mutationFn: api.createStream, onSuccess: () => { setAdding(false); setForm({ stream_id: "", name: "", source: "fake" }); client.invalidateQueries({ queryKey: ["streams"] }); } });
  const remove = useMutation({ mutationFn: api.deleteStream, onSuccess: () => client.invalidateQueries({ queryKey: ["streams"] }) });
  return <div className="page-content">
    <div className="page-heading"><div><p className="eyebrow">MANAGE / STREAMS</p><h1>Camera streams</h1><p className="page-subtitle">Manage the cameras connected to your retail locations.</p></div><Button onClick={() => setAdding((value) => !value)}><CirclePlus size={16} /> Add stream</Button></div>
    {query.isError && query.error instanceof Error && <ErrorState error={query.error} retry={() => query.refetch()} />}
    {adding && <Card className="add-stream"><CardHeading title="Add a stream" detail="Create a camera configuration" /><form onSubmit={(event) => { event.preventDefault(); create.mutate({ ...form, autostart: false }); }}><label>Stream ID<input required pattern="[\\w.-]+" value={form.stream_id} onChange={(event) => setForm({ ...form, stream_id: event.target.value })} placeholder="store-camera-01" /></label><label>Display name<input required value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="North entrance" /></label><label>Source<input required value={form.source} onChange={(event) => setForm({ ...form, source: event.target.value })} placeholder="fake, camera index, video path, or RTSP URL" /></label><div><Button type="submit" disabled={create.isPending}>Create stream</Button><Button type="button" variant="quiet" onClick={() => setAdding(false)}>Cancel</Button></div>{create.error instanceof Error && <p className="form-error">{create.error.message}</p>}</form></Card>}
    <div className="stream-summary"><div><span className="summary-icon"><Camera size={18} /></span><span><strong>{query.data?.length ?? 0}</strong><small>Total cameras</small></span></div><div><span className="summary-icon online"><span /></span><span><strong>{query.data?.filter((item) => item.status === "RUNNING").length ?? 0}</strong><small>Active streams</small></span></div><span className="muted-small">API SYNCHRONIZED</span></div>
    <Card className="streams-table-card"><CardHeading title="All streams" detail="Stream status and configuration" action={<span className="muted-small">REFRESHES AUTOMATICALLY</span>} />
      <div className="table-scroll"><table><thead><tr><th>STREAM</th><th>SOURCE</th><th>STATUS</th><th>FRAME RATE</th><th>LATENCY</th><th>ACTIONS</th></tr></thead><tbody>{query.isLoading ? Array.from({ length: 3 }, (_, i) => <tr key={i}><td colSpan={6}><Skeleton className="row-skeleton" /></td></tr>) : (query.data ?? []).map((stream: Stream) => <tr key={stream.stream_id}><td><button className="stream-name" onClick={() => selectStream(stream.stream_id)}><span className="camera-icon"><Camera size={17} /></span><span><strong>{stream.name}</strong><small>{stream.stream_id}</small></span></button></td><td>{stream.source}</td><td><Badge tone={stream.status === "RUNNING" ? "good" : "neutral"}>{stream.status === "RUNNING" ? "Online" : "Stopped"}</Badge></td><td>{Number(stream.metrics?.output_fps ?? 0).toFixed(1)} fps</td><td>{Number(stream.metrics?.latency_p95_ms ?? 0)} ms</td><td><div className="table-actions">{stream.status === "RUNNING" ? <Button title="Stop stream" variant="quiet" onClick={() => lifecycle.mutate({ id: stream.stream_id, action: "stop" })}><Square size={15} /></Button> : <Button title="Start stream" variant="quiet" onClick={() => lifecycle.mutate({ id: stream.stream_id, action: "start" })}><Play size={15} /></Button>}<Button title="Restart stream" variant="quiet" onClick={() => lifecycle.mutate({ id: stream.stream_id, action: "restart" })}><RotateCw size={15} /></Button><Button title="Delete stream" variant="quiet" onClick={() => { if (window.confirm(`Delete ${stream.name}?`)) remove.mutate(stream.stream_id); }}><Trash2 size={15} /></Button></div></td></tr>)}</tbody></table>{query.data?.length === 0 && <EmptyState title="No streams added" text="Add a stream to begin monitoring your store." />}</div>
    </Card>
  </div>;
}
