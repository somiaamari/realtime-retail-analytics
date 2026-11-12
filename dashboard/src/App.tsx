import { Suspense, lazy, useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { Activity, BarChart3, Camera, ChevronDown, ChevronLeft, ChevronRight, KeyRound, LayoutDashboard, Menu, Moon, Sun, Waypoints, X } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api/client";
import { useDashboardStore } from "./lib/store";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Skeleton } from "./components/ui";

const LivePage = lazy(() => import("./pages/LivePage"));
const AnalyticsPage = lazy(() => import("./pages/AnalyticsPage"));
const ZonesPage = lazy(() => import("./pages/ZonesPage"));
const StreamsPage = lazy(() => import("./pages/StreamsPage"));
const SystemPage = lazy(() => import("./pages/SystemPage"));

const navItems = [
  { to: "/", label: "Live overview", icon: LayoutDashboard },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/zones", label: "Zones & lines", icon: Waypoints },
  { to: "/streams", label: "Streams", icon: Camera },
  { to: "/system", label: "System health", icon: Activity },
];

const routeTitles: Record<string, string> = { "/": "Live overview", "/analytics": "Analytics", "/zones": "Zones & lines", "/streams": "Streams", "/system": "System health" };

export default function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const path = useLocation().pathname;
  const { theme, toggleTheme, streamId, setStreamId, range, setRange } = useDashboardStore();
  const streamQuery = useQuery({ queryKey: ["streams"], queryFn: api.streams, refetchInterval: 15000 });
  const streams = streamQuery.data ?? [];
  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  const mockMode = import.meta.env.VITE_USE_MOCKS === "true";

  return <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
    <aside className={`sidebar ${drawer ? "drawer-open" : ""}`}>
      <div className="brand"><div className="brand-mark"><span /></div>{!collapsed && <span>retail<span className="brand-light">vision</span></span>}<button className="sidebar-close" aria-label="Close navigation" onClick={() => setDrawer(false)}><X size={18} /></button></div>
      <div className="workspace-label">{!collapsed && "WORKSPACE"}</div>
      <nav aria-label="Main navigation">{navItems.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} end={to === "/"} title={collapsed ? label : undefined} onClick={() => setDrawer(false)} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}><Icon size={18} /><span>{label}</span>{to === "/" && <i className="nav-live-dot" />}</NavLink>)}</nav>
      <div className="sidebar-bottom"><div className="store-card"><span className="store-avatar">N</span>{!collapsed && <span><strong>Northstar Retail</strong><small>Pro workspace</small></span>}<ChevronDown size={14} /></div><button className="collapse-button" onClick={() => setCollapsed((value) => !value)} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>{collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}{!collapsed && "Collapse menu"}</button></div>
    </aside>
    {drawer && <button className="drawer-backdrop" onClick={() => setDrawer(false)} aria-label="Close navigation overlay" />}
    <div className="app-main">
      <header className="topbar"><button className="mobile-menu" aria-label="Open navigation" onClick={() => setDrawer(true)}><Menu size={19} /></button><div className="breadcrumb"><span>Workspace</span><span>/</span><strong>{routeTitles[path] ?? "Analytics"}</strong></div><div className="topbar-tools">
        <label className="toolbar-select stream-select"><span className="selector-dot" /><select aria-label="Select stream" value={streamId} onChange={(event) => setStreamId(event.target.value)}>{streams.map((stream) => <option key={stream.stream_id} value={stream.stream_id}>{stream.name}</option>)}{!streams.length && <option value="demo-store">Demo store</option>}</select><ChevronDown size={14} /></label>
        <label className="toolbar-select range-select"><select aria-label="Global date range" value={range} onChange={(event) => setRange(Number(event.target.value))}><option value={24}>Last 24 hours</option><option value={168}>Last 7 days</option><option value={720}>Last 30 days</option></select><ChevronDown size={14} /></label>
        <span className={`connection-pill ${mockMode ? "mock-pill" : ""}`}><i />{mockMode ? "DEMO MODE" : "API CONNECTED"}</span>
        {!mockMode && <button className="theme-button" onClick={() => { const current = sessionStorage.getItem("rv-api-key") ?? ""; const key = window.prompt("API key (kept in this browser session only)", current); if (key === null) return; if (key.trim()) sessionStorage.setItem("rv-api-key", key.trim()); else sessionStorage.removeItem("rv-api-key"); }} aria-label="Configure API key for this session" title="Configure API key"><KeyRound size={15} /></button>}
        <button className="theme-button" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>{theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}</button>
      </div></header>
      <main className="main-content"><ErrorBoundary><Suspense fallback={<div className="page-content"><Skeleton className="title-skeleton" /><div className="kpi-grid">{Array.from({ length: 4 }, (_, i) => <Skeleton className="kpi-skeleton" key={i} />)}</div><Skeleton className="content-skeleton" /></div>}><Routes><Route path="/" element={<LivePage />} /><Route path="/analytics" element={<AnalyticsPage />} /><Route path="/zones" element={<ZonesPage />} /><Route path="/streams" element={<StreamsPage />} /><Route path="/system" element={<SystemPage />} /><Route path="*" element={<LivePage />} /></Routes></Suspense></ErrorBoundary></main>
      <footer className="footer"><span>Retail Vision Analytics</span><span>Privacy-first · No face identification</span><span className="footer-version">DASHBOARD · 0.1.0</span></footer>
    </div>
  </div>;
}
