import { describe, expect, it } from "vitest";
import { parseSocketMessage } from "./protocol";

describe("WebSocket protocol", () => {
  it("accepts known server envelopes", () => {
    expect(parseSocketMessage('{"type":"SNAPSHOT","data":{"occupancy":{"main":3}}}')).toMatchObject({ type: "SNAPSHOT" });
  });
  it("rejects malformed or unknown messages", () => {
    expect(parseSocketMessage("{broken")).toBeNull();
    expect(parseSocketMessage('{"type":"UNEXPECTED"}')).toBeNull();
  });
});
