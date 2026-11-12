import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { beforeAll, afterAll, afterEach, describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import AnalyticsPage from "./AnalyticsPage";
import { handlers } from "../mocks/handlers";

const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => { server.resetHandlers(); });
afterAll(() => server.close());

function renderAnalytics() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}><MemoryRouter><AnalyticsPage /></MemoryRouter></QueryClientProvider>);
}

describe("Analytics page", () => {
  it("renders analytics cards and fetched event rows", async () => {
    renderAnalytics();
    expect(screen.getByRole("heading", { name: "Store analytics" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("ZONE ENTER").length).toBeGreaterThan(0));
    expect(screen.getByRole("img", { name: "Peak traffic by day of week and hour" })).toBeInTheDocument();
  });

  it("surfaces API errors accessibly", async () => {
    server.use(http.get("/api/v1/streams/:id/analytics/summary", () => HttpResponse.json({ detail: "unavailable" }, { status: 503 })));
    renderAnalytics();
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
  });
});
