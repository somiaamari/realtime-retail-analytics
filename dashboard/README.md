# Retail Vision dashboard

The dashboard is a React 18 + TypeScript SPA for live store monitoring and historical retail analytics. It can connect to the FastAPI service or run as a self-contained static demo.

## Requirements and scripts

Node.js 20+ and npm 10+.

```shell
npm ci
npm run gen:api       # regenerate src/api/schema.d.ts from ../docs/openapi.json
npm run dev           # local development server at http://localhost:5173
npm run typecheck
npm run lint
npm test
npm run build
npm run test:e2e
```

The Vite development server proxies `/api` and `/ws` to `http://localhost:8000`. Set `VITE_API_PROXY` if the backend is elsewhere.

## Real backend

Start the backend as described in [Phase 9 API documentation](../docs/phase-09-api.md), then copy `.env.example` to `.env.local` and set:

| Variable | Description |
|---|---|
| `VITE_USE_MOCKS` | Set to `false` for the real API. |
| `VITE_API_BASE_URL` | Optional API origin. Leave empty to use the same origin and the Vite proxy in development. |
| `VITE_API_PROXY` | Development-only Vite proxy target, default `http://localhost:8000`. |

If the backend requires an API key, set it from the key icon in the top bar; it is held in browser session storage only and sent over HTTPS for REST (and as the WebSocket query parameter required by the backend). Do not place API keys in Vite environment variables or public builds. Only streams that the backend has configured and started can provide live snapshots/video. The video image is the annotated MJPEG endpoint; the dashboard does not display or identify faces.

## Mock mode

```shell
npm run dev -- --mode mock
```

The checked-in `.env.mock` enables MSW REST handlers and a simulated WebSocket feed. The live video is rendered locally on a canvas, so the static demo has no backend dependency. Mock state and API shapes are intentionally aligned with the committed OpenAPI contract. `npm run test:e2e` starts this mode automatically.

## Themes and design tokens

The single token layer is `src/styles/tokens.css`; it defines palette, semantic colors, surfaces, borders, and theme variants. The requested palette is used for brand surfaces and chart accents: `#202734`, `#36415A`, `#B7D894`, and `#E0F8C7`. Toggle themes from the top bar. Reduced-motion preferences, keyboard focus rings, chart descriptions, and mobile layouts are supported.

## Static hosting (Netlify)

`netlify.toml` configures the `dashboard` base directory, SPA route fallback, and production build. Set `VITE_USE_MOCKS=true` in the Netlify environment before deploying, or use `.env.mock` locally to preview the same build:

```shell
npm run build -- --mode mock
npm run preview
```

## Architecture

- `src/api`: API client, shared models, and generated OpenAPI types.
- `src/realtime`: validated WebSocket envelopes, reconnection, visibility handling, and update batching.
- `src/mocks`: MSW REST fixtures and the deterministic mock socket.
- `src/pages`: lazy-loaded Live, Analytics, Zones, Streams, and System routes.
- `src/components`: design primitives, video, and Canvas heatmap.
- `src/lib`: Zustand state, formatting, and normalized geometry utilities.
