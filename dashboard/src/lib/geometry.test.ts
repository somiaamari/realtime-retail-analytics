import { describe, expect, it } from "vitest";
import { hasSelfIntersection, pointInPolygon } from "./geometry";

describe("geometry", () => {
  it("detects points inside and outside a normalized zone", () => {
    const square: [number, number][] = [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1, 0.8]];
    expect(pointInPolygon([0.5, 0.5], square)).toBe(true);
    expect(pointInPolygon([0.9, 0.5], square)).toBe(false);
  });

  it("flags crossing polygon edges", () => {
    expect(hasSelfIntersection([[0, 0], [1, 1], [0, 1], [1, 0]])).toBe(true);
    expect(hasSelfIntersection([[0, 0], [1, 0], [1, 1], [0, 1]])).toBe(false);
  });
});
