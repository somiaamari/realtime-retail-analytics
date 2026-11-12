import { useEffect, useRef } from "react";

const ramp = ["#202734", "#36415A", "#71885E", "#B7D894", "#E0F8C7"];

export function HeatmapCanvas({ values, label = "Floor traffic heatmap" }: { values: number[][]; label?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || values.length === 0) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const width = values[0]?.length ?? 0;
    const height = values.length;
    canvas.width = width;
    canvas.height = height;
    const image = context.createImageData(width, height);
    const max = Math.max(1, ...values.flat());
    const colors = ramp.map((hex) => hex.match(/\w\w/g)!.map((part) => parseInt(part, 16)));
    values.forEach((row, y) => row.forEach((value, x) => {
      const ratio = Math.min(0.999, Math.max(0, value / max)) * (colors.length - 1);
      const index = Math.floor(ratio);
      const blend = ratio - index;
      const from = colors[index];
      const to = colors[index + 1];
      const offset = (y * width + x) * 4;
      image.data[offset] = from[0] + (to[0] - from[0]) * blend;
      image.data[offset + 1] = from[1] + (to[1] - from[1]) * blend;
      image.data[offset + 2] = from[2] + (to[2] - from[2]) * blend;
      image.data[offset + 3] = 235;
    }));
    context.putImageData(image, 0, 0);
  }, [values]);
  return <canvas ref={ref} className="heatmap-canvas" role="img" aria-label={label} />;
}
