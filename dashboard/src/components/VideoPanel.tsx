import { useEffect, useRef } from "react";
import { Download, Expand, Radio } from "lucide-react";
import { api } from "../api/client";
import { useDashboardStore } from "../lib/store";
import { Card, CardHeading, Skeleton } from "./ui";

const mock = import.meta.env.VITE_USE_MOCKS === "true";

function MockVideo() {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext("2d");
    if (!element || !context) return;
    let frame = 0;
    let animation = 0;
    const draw = () => {
      frame += 0.012;
      const gradient = context.createLinearGradient(0, 0, element.width, element.height);
      gradient.addColorStop(0, "#263143");
      gradient.addColorStop(1, "#101820");
      context.fillStyle = gradient;
      context.fillRect(0, 0, element.width, element.height);
      context.strokeStyle = "rgba(183,216,148,.17)";
      context.lineWidth = 1;
      for (let x = 0; x < element.width; x += 48) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, element.height); context.stroke(); }
      for (let y = 0; y < element.height; y += 42) { context.beginPath(); context.moveTo(0, y); context.lineTo(element.width, y); context.stroke(); }
      const people = [[.2, .54], [.38, .44], [.59, .63], [.77, .39], [.48, .3]];
      people.forEach(([x, y], i) => {
        const px = (x + Math.sin(frame + i) * .025) * element.width;
        const py = y * element.height;
        context.strokeStyle = "#B7D894";
        context.strokeRect(px - 21, py - 52, 42, 66);
        context.fillStyle = "#E0F8C7";
        context.font = "11px sans-serif";
        context.fillText(`visitor ${i + 1}`, px - 25, py - 57);
        context.fillStyle = "rgba(183,216,148,.25)";
        context.beginPath(); context.arc(px, py - 10, 26, 0, Math.PI * 2); context.fill();
      });
      animation = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(animation);
  }, []);
  return <canvas ref={canvas} width={960} height={540} className="video-image" aria-label="Simulated annotated store video" />;
}

export function VideoPanel({ online }: { online: boolean }) {
  const streamId = useDashboardStore((state) => state.streamId);
  return <Card className="video-card">
    <CardHeading title="Live camera" detail="Annotated feed · privacy-safe" action={<span className={`live-indicator ${online ? "is-live" : ""}`}><span />{online ? "LIVE" : "OFFLINE"}</span>} />
    <div className="video-frame">
      {online ? (mock ? <MockVideo /> : <img className="video-image" src={api.videoUrl(streamId)} alt="Live annotated camera feed. Faces are not displayed." />) : <div className="video-offline"><Radio size={30} /><strong>Stream is offline</strong><span>Start the stream to see its live feed.</span></div>}
      {online && <div className="video-overlay"><span>CAM 01 · NORTH ENTRANCE</span><span>ANALYTICS ACTIVE</span></div>}
      <button className="video-fullscreen" aria-label="Open video fullscreen" onClick={(event) => event.currentTarget.parentElement?.requestFullscreen()}><Expand size={16} /></button>
    </div>
    <div className="video-footer"><span><span className="status-led" />{online ? "Signal stable · 1080p" : "No signal"}</span>{!mock && <a className="text-action" href={api.snapshotUrl(streamId)} download={`${streamId}-snapshot.jpg`}><Download size={15} /> Snapshot</a>}</div>
    {!online && <Skeleton className="video-skeleton" />}
  </Card>;
}
