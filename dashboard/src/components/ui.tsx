import { useEffect, useRef, type ButtonHTMLAttributes, type PropsWithChildren, type ReactNode } from "react";
import { AlertCircle, ArrowDownRight, ArrowUpRight, LoaderCircle, X } from "lucide-react";

export function Button({ children, variant = "primary", ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "quiet" | "outline" | "danger" }) {
  return <button className={`button button-${variant}`} {...props}>{children}</button>;
}

export function Card({ children, className = "" }: PropsWithChildren<{ className?: string }>) {
  return <section className={`card ${className}`}>{children}</section>;
}

export function CardHeading({ title, detail, action }: { title: string; detail?: string; action?: ReactNode }) {
  return <div className="card-heading"><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{action}</div>;
}

export function KpiCard({ label, value, delta, icon, tone = "mint" }: { label: string; value: string | number; delta?: string; icon: ReactNode; tone?: "mint" | "blue" | "amber" | "rose" }) {
  return <Card className="kpi-card"><div className={`kpi-icon ${tone}`}>{icon}</div><div className="kpi-label">{label}</div><strong className="kpi-value">{value}</strong>{delta && <span className="kpi-delta"><ArrowUpRight size={14} />{delta}</span>}</Card>;
}

export function Badge({ children, tone = "neutral" }: PropsWithChildren<{ tone?: "good" | "warning" | "neutral" | "mint" }>) {
  return <span className={`badge badge-${tone}`}><span className="badge-dot" />{children}</span>;
}

export function EmptyState({ title, text }: { title: string; text: string }) {
  return <div className="empty-state"><div className="empty-mark">—</div><strong>{title}</strong><p>{text}</p></div>;
}

export function ErrorState({ error, retry }: { error: Error; retry?: () => void }) {
  return <div className="error-state" role="alert"><AlertCircle size={20} /><div><strong>Couldn’t load this data</strong><p>{error.message}</p>{retry && <Button variant="outline" onClick={retry}>Try again</Button>}</div></div>;
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-label="Loading" />;
}

export function LoadingMark() {
  return <LoaderCircle className="spin" size={16} aria-label="Loading" />;
}

export function Delta({ value }: { value: number }) {
  return <span className={`delta ${value < 0 ? "negative" : ""}`}>{value < 0 ? <ArrowDownRight size={13} /> : <ArrowUpRight size={13} />}{Math.abs(value).toFixed(1)}%</span>;
}

export function Dialog({ open, title, onClose, children }: PropsWithChildren<{ open: boolean; title: string; onClose: () => void }>) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement;
    ref.current?.focus();
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("keydown", onKey); if (previous instanceof HTMLElement) previous.focus(); };
  }, [open, onClose]);
  if (!open) return null;
  return <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}><section ref={ref} className="dialog" role="dialog" aria-modal="true" aria-labelledby="dialog-title" tabIndex={-1}><header><h2 id="dialog-title">{title}</h2><button aria-label="Close dialog" onClick={onClose}><X size={16} /></button></header>{children}</section></div>;
}

export function Tabs({ value, options, onChange }: { value: string; options: string[]; onChange: (value: string) => void }) {
  return <div className="tabs" role="tablist">{options.map((option) => <button key={option} role="tab" aria-selected={value === option} className={value === option ? "active" : ""} onClick={() => onChange(option)}>{option}</button>)}</div>;
}

export function Select({ label, value, options, onChange }: { label: string; value: string; options: Array<{ value: string; label: string }>; onChange: (value: string) => void }) {
  return <label className="field-select"><span>{label}</span><select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}>{options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>;
}

export function Tooltip({ label, children }: PropsWithChildren<{ label: string }>) {
  return <span className="tooltip" tabIndex={0} aria-label={label}><span className="tooltip-content" role="tooltip">{label}</span>{children}</span>;
}

export function Switch({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return <button className={`switch ${checked ? "checked" : ""}`} type="button" role="switch" aria-checked={checked} aria-label={label} onClick={() => onChange(!checked)}><span /></button>;
}

export function Slider({ label, value, min = 0, max = 100, onChange }: { label: string; value: number; min?: number; max?: number; onChange: (value: number) => void }) {
  return <label className="slider"><span>{label}</span><input type="range" aria-label={label} min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} /><output>{value}</output></label>;
}

export function Toast({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return <div className="toast" role="status"><span>{message}</span><button aria-label="Dismiss notification" onClick={onDismiss}><X size={14} /></button></div>;
}
