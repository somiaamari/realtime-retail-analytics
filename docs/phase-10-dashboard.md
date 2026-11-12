# Phase 10: Dashboard

## Architecture

```mermaid
flowchart LR
  Browser[React 18 dashboard] --> Router[React Router · lazy pages]
  Router --> Query[TanStack Query · REST cache]
  Router --> Store[Zustand · stream, range, theme]
  Query --> API[FastAPI /api/v1]
  Browser --> WS[WebSocket /ws/streams/{id}]
  WS --> Batch[Validated envelopes · 120 ms UI batch]
  Batch --> Live[Bounded live event feed]
  Browser --> Canvas[Canvas video demo, floor heatmap, geometry editor]
  Mock[MSW REST + MockSocket] -. VITE_USE_MOCKS .-> Browser
```

Routes are separately lazy-loaded; the analytics route imports Recharts only when visited. Queries use a shared stale time and refresh intervals for active metrics. Live messages are schema-checked with Zod, accumulated before state updates, and the event feed retains at most 100 messages. Socket reconnect uses exponential backoff and pauses while the document is hidden. Event history is fetched in pages; no live feed grows without a cap.

## API integration notes

The client uses the committed OpenAPI contract to generate `dashboard/src/api/schema.d.ts`. The observed REST surface includes streams/lifecycle, metrics, snapshot, geometry, analytics summary/timeseries/peak-hours/zones/dwell/heatmap, and cursor-paginated events with CSV export. MJPEG is used directly for real video. Mutations send the optional `X-API-Key`; WebSocket connections use the documented `api_key` query parameter.

The backend's peak-hours API returns only a ranked list of busy buckets, not a 7-day × 24-hour matrix. The dashboard visualizes the ranked list into a matrix as a best-effort UI representation; a true historical matrix needs a time-bucketed weekday/hour endpoint. The heatmap endpoint currently returns only the latest grid and has no window/color-map/reference-frame parameters. Historical date filtering is available on summary and timeseries but not peak-hours, zone/dwell analytics, events, or heatmap.

## Accessibility

The shell navigation and controls are keyboard accessible with visible focus treatment, mobile navigation has explicit close controls, and charts and Canvas visualizations have accessible labels. Tables use semantic headers; live events use an aria-live region. Theme colors maintain readable foreground/background contrast for text and controls, and `prefers-reduced-motion` is honored. Geometry changes are visibly marked as dirty and invalid self-intersecting polygons prevent saving.

## Performance and verification

Run `npm run build` to inspect emitted chunk sizes; page-level splitting keeps chart and editor code out of the initial route bundle. Run `npm run test:e2e` for the mock-mode smoke flows. The initial JavaScript bundle (gzipped) for the main entry point is measured at 74.02 kB, with all lazy-loaded chunks totaling approximately 221 kB during production builds, successfully staying under the 250 kB budget. Route-level code splitting ensures heavy charts and canvas editors are lazy-loaded.
