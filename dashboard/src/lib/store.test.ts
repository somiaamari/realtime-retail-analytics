import { beforeEach, describe, expect, it } from "vitest";
import { rangeBounds, useDashboardStore } from "./store";

describe("dashboard store", () => {
  beforeEach(() => { useDashboardStore.setState({ streamId: "demo-store", range: 24 }); });
  it("updates global stream and date range", () => {
    useDashboardStore.getState().setStreamId("front-door");
    useDashboardStore.getState().setRange(168);
    expect(useDashboardStore.getState()).toMatchObject({ streamId: "front-door", range: 168 });
  });
  it("calculates a selected range", () => {
    const [start, end] = rangeBounds(24);
    expect(new Date(end).getTime() - new Date(start).getTime()).toBe(86_400_000);
  });
});
