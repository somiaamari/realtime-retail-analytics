import { describe, expect, it } from "vitest";
import type { components } from "./schema";

describe("OpenAPI mock contracts", () => {
  it("keeps representative request fixtures compatible with generated API schemas", () => {
    const stream: components["schemas"]["StreamCreate"] = {
      stream_id: "demo-store", name: "Demo store", source: "fake", backend: "pytorch",
      zones: [], lines: [], detect_every_n: 1, autostart: false,
    };
    const geometry: components["schemas"]["GeometryUpdate"] = {
      zones: [{ id: "entrance", name: "Entrance", normalized: true, points: [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]] }],
      lines: [],
    };
    expect(stream.stream_id).toBe("demo-store");
    expect(geometry.zones).toHaveLength(1);
  });
});
