import { useEffect, useRef, useState, type PointerEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Crosshair, MousePointer2, Plus, RotateCcw, Save, Trash2 } from "lucide-react";
import type { Geometry } from "../api/types";
import { api } from "../api/client";
import { useDashboardStore } from "../lib/store";
import { hasSelfIntersection, type Point } from "../lib/geometry";
import { Button, Card, CardHeading, EmptyState } from "../components/ui";

type Tool = "select" | "zone" | "line";
const initialGeometry: Geometry = { zones: [], lines: [] };

export default function ZonesPage() {
  const streamId = useDashboardStore((state) => state.streamId);
  const client = useQueryClient();
  const query = useQuery({ queryKey: ["geometry", streamId], queryFn: () => api.geometry(streamId) });
  const [geometry, setGeometry] = useState<Geometry>(initialGeometry);
  const [tool, setTool] = useState<Tool>("select");
  const [draft, setDraft] = useState<Point[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [dragTarget, setDragTarget] = useState<{ zoneId: string; pointIndex: number } | null>(null);
  const [saving, setSaving] = useState(false);
  const canvas = useRef<HTMLCanvasElement>(null);
  const dirty = JSON.stringify(geometry) !== JSON.stringify(query.data ?? initialGeometry);

  useEffect(() => { if (query.data) setGeometry(query.data); }, [query.data]);
  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext("2d");
    if (!element || !context) return;
    const { width, height } = element;
    context.fillStyle = "#19212b";
    context.fillRect(0, 0, width, height);
    context.strokeStyle = "rgba(224,248,199,.07)";
    for (let x = 0; x < width; x += 42) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
    for (let y = 0; y < height; y += 42) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
    context.fillStyle = "rgba(255,255,255,.025)";
    context.fillRect(width * .36, 0, 4, height);
    geometry.zones.forEach((zone) => {
      context.beginPath();
      zone.points.forEach(([x, y], i) => i ? context.lineTo(x * width, y * height) : context.moveTo(x * width, y * height));
      context.closePath();
      context.fillStyle = zone.id === selected ? "rgba(183,216,148,.22)" : "rgba(183,216,148,.12)";
      context.strokeStyle = "#B7D894";
      context.lineWidth = zone.id === selected ? 2.5 : 1.5;
      context.fill(); context.stroke();
      context.fillStyle = "#E0F8C7";
      context.font = "13px sans-serif";
      context.fillText(zone.name, zone.points[0][0] * width + 8, zone.points[0][1] * height + 18);
      zone.points.forEach(([x, y]) => { context.beginPath(); context.arc(x * width, y * height, 4, 0, Math.PI * 2); context.fillStyle = "#E0F8C7"; context.fill(); });
    });
    geometry.lines.forEach((line) => {
      context.beginPath();
      line.points.forEach(([x, y], i) => i ? context.lineTo(x * width, y * height) : context.moveTo(x * width, y * height));
      context.strokeStyle = "#8fa9d8"; context.lineWidth = 3; context.stroke();
    });
    if (draft.length) {
      context.beginPath();
      draft.forEach(([x, y], i) => i ? context.lineTo(x * width, y * height) : context.moveTo(x * width, y * height));
      context.strokeStyle = "#E0F8C7"; context.setLineDash([5, 5]); context.stroke(); context.setLineDash([]);
    }
  }, [geometry, draft, selected]);

  function canvasPoint(event: React.MouseEvent<HTMLCanvasElement>): Point {
    const rect = event.currentTarget.getBoundingClientRect();
    return [(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height];
  }
  function handleCanvasClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const point = canvasPoint(event);
    if (tool !== "select") {
      const next = [...draft, point];
      setDraft(next);
      const done = (tool === "zone" && next.length >= 6) || (tool === "line" && next.length >= 2);
      if (done) {
        const id = `${tool}-${Date.now()}`;
        if (tool === "zone") setGeometry((value) => ({ ...value, zones: [...value.zones, { id, name: "New zone", normalized: true, points: next }] }));
        else setGeometry((value) => ({ ...value, lines: [...value.lines, { id, name: "New line", normalized: true, points: next.slice(0, 2) }] }));
        setSelected(id); setDraft([]);
      }
      return;
    }
    const match = geometry.zones.find((zone) => zone.points.some(([x, y]) => Math.hypot(point[0] - x, point[1] - y) < .035));
    if (match) setSelected(match.id);
  }
  function handlePointerDown(event: PointerEvent<HTMLCanvasElement>) {
    if (tool !== "select") return;
    const point = canvasPoint(event);
    for (const zone of geometry.zones) {
      const pointIndex = zone.points.findIndex(([x, y]) => Math.hypot(point[0] - x, point[1] - y) < .035);
      if (pointIndex >= 0) {
        event.currentTarget.setPointerCapture(event.pointerId);
        setSelected(zone.id);
        setDragTarget({ zoneId: zone.id, pointIndex });
        return;
      }
    }
  }
  function handlePointerMove(event: PointerEvent<HTMLCanvasElement>) {
    if (!dragTarget) return;
    const [x, y] = canvasPoint(event);
    setGeometry((value) => ({
      ...value,
      zones: value.zones.map((zone) => zone.id === dragTarget.zoneId
        ? { ...zone, points: zone.points.map((point, index) => index === dragTarget.pointIndex ? [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))] : point) }
        : zone),
    }));
  }
  function renameSelected() {
    const current = geometry.zones.find((zone) => zone.id === selected) ?? geometry.lines.find((line) => line.id === selected);
    if (!current) return;
    const name = window.prompt("Geometry name", current.name);
    if (!name?.trim()) return;
    setGeometry((value) => ({ zones: value.zones.map((zone) => zone.id === selected ? { ...zone, name: name.trim() } : zone), lines: value.lines.map((line) => line.id === selected ? { ...line, name: name.trim() } : line) }));
  }
  function deleteSelected() {
    setGeometry((value) => ({ zones: value.zones.filter((zone) => zone.id !== selected), lines: value.lines.filter((line) => line.id !== selected) }));
    setSelected(null);
  }
  async function save() {
    setSaving(true);
    try { await api.saveGeometry(streamId, geometry); await client.invalidateQueries({ queryKey: ["geometry", streamId] }); }
    finally { setSaving(false); }
  }
  function discard() { setGeometry(query.data ?? initialGeometry); setDraft([]); setSelected(null); }
  const invalid = geometry.zones.some((zone) => hasSelfIntersection(zone.points as Point[]));

  return <div className="page-content">
    <div className="page-heading"><div><p className="eyebrow">CONFIGURATION / GEOMETRY</p><h1>Zones & lines</h1><p className="page-subtitle">Draw normalized areas and counting lines on your camera view.</p></div><div className="heading-actions"><Button variant="outline" disabled={!dirty} onClick={discard}><RotateCcw size={15} /> Discard</Button><Button disabled={!dirty || invalid || saving} onClick={save}><Save size={15} /> {saving ? "Saving…" : "Save changes"}</Button></div></div>
    <div className="editor-grid">
      <Card className="editor-card"><CardHeading title="Camera reference" detail="Coordinates are stored proportionally (0–1)" action={<span className="muted-small">{tool === "select" ? "SELECT & DRAG" : `DRAW ${tool.toUpperCase()} · DOUBLE-CLICK TO FINISH`}</span>} />
        <div className="editor-toolbar"><Button variant={tool === "select" ? "primary" : "quiet"} onClick={() => setTool("select")}><MousePointer2 size={15} /> Select</Button><Button variant={tool === "zone" ? "primary" : "quiet"} onClick={() => { setTool("zone"); setDraft([]); }}><Plus size={15} /> Draw zone</Button><Button variant={tool === "line" ? "primary" : "quiet"} onClick={() => { setTool("line"); setDraft([]); }}><Crosshair size={15} /> Counting line</Button><span className="toolbar-spacer" />{selected && <Button variant="quiet" onClick={renameSelected}>Rename</Button>}{selected && <Button variant="quiet" onClick={deleteSelected}><Trash2 size={15} /></Button>}</div>
        <div className="editor-canvas-wrap"><canvas ref={canvas} width={960} height={540} className="editor-canvas" onClick={handleCanvasClick} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={() => setDragTarget(null)} onPointerCancel={() => setDragTarget(null)} onDoubleClick={(event) => { if (tool === "zone" && draft.length >= 2) { event.preventDefault(); const points = [...draft, canvasPoint(event)]; const id = `zone-${Date.now()}`; setGeometry((value) => ({ ...value, zones: [...value.zones, { id, name: "New zone", normalized: true, points }] })); setDraft([]); setSelected(id); } }} aria-label="Interactive zone and line editor" role="application" tabIndex={0} /></div>
        <div className="editor-legend"><span><i className="legend-dot green" />Zone</span><span><i className="legend-dot blue" />Counting line</span><span>Click points to draw · double-click to complete a zone</span></div>
      </Card>
      <div className="editor-side">
        <Card><CardHeading title="Geometry" detail={`${geometry.zones.length} zones · ${geometry.lines.length} lines`} />
          {geometry.zones.some((zone) => zone.id === selected) && <label className="geometry-type">Selected zone type<select aria-label="Selected zone type" value={String(geometry.zones.find((zone) => zone.id === selected)?.type ?? "occupancy")} onChange={(event) => setGeometry((value) => ({ ...value, zones: value.zones.map((zone) => zone.id === selected ? { ...zone, type: event.target.value } : zone) }))}><option value="occupancy">Occupancy</option><option value="dwell">Dwell</option><option value="restricted">Restricted area</option></select></label>}
          <div className="geometry-list">{[...geometry.zones.map((item) => ({ ...item, kind: "Zone" })), ...geometry.lines.map((item) => ({ ...item, kind: "Line" }))].map((item) => <button className={`geometry-item ${item.id === selected ? "selected" : ""}`} key={item.id} onClick={() => setSelected(item.id)}><span className={`geometry-glyph ${item.kind.toLowerCase()}`} /><span><strong>{item.name}</strong><small>{item.kind} · {item.points.length} points</small></span><span className="geometry-count"><Check size={14} /></span></button>)}</div>
          {!geometry.zones.length && !geometry.lines.length && <EmptyState title="No geometry configured" text="Add a zone or counting line to begin." />}
        </Card>
        <Card className="editor-help"><span className="tip-label">EDITOR TIP</span><h3>Map your store floor</h3><p>Draw around high-traffic areas to measure dwell time and occupancy. Add lines across entrances to count visits.</p><div className="tip-footer"><span className="status-led" />Changes are saved to this stream</div></Card>
        {invalid && <div className="error-state" role="alert">Zone boundaries cannot cross themselves.</div>}
        {!dirty && <div className="saved-note"><Check size={14} /> All changes saved</div>}
      </div>
    </div>
  </div>;
}
