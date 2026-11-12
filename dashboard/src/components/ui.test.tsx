import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { describe, expect, it } from "vitest";
import { KpiCard } from "./ui";

describe("KpiCard", () => {
  it("renders a named metric and trend", () => {
    render(<KpiCard label="People now" value={34} delta="+8% this hour" icon={<span aria-hidden="true">●</span>} />);
    expect(screen.getByText("People now")).toBeInTheDocument();
    expect(screen.getByText("34")).toBeInTheDocument();
    expect(screen.getByText("+8% this hour")).toBeInTheDocument();
  });

  it("has no basic accessibility violations", async () => {
    const { container } = render(<KpiCard label="People now" value={34} icon={<span aria-hidden="true">●</span>} />);
    const results = await axe(container);
    expect(results.violations).toHaveLength(0);
  });
});
