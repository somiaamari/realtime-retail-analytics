import { describe, expect, it } from "vitest";
import { formatDuration, formatNumber, formatPercent } from "./format";

describe("formatters", () => {
  it("formats metrics for dashboard cards", () => {
    expect(formatNumber(1204)).toBe("1,204");
    expect(formatDuration(286)).toBe("4m 46s");
    expect(formatPercent(-2.3)).toBe("-2.3%");
  });
});
