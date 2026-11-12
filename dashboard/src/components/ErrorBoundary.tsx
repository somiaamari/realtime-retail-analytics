import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props { children: ReactNode }
interface State { error: Error | null }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };
  static getDerivedStateFromError(error: Error): State { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo): void { console.error("Page render failed", error, info.componentStack); }
  render(): ReactNode {
    if (this.state.error) return <div className="page-error" role="alert"><h2>Page unavailable</h2><p>{this.state.error.message}</p><button className="button button-outline" onClick={() => this.setState({ error: null })}>Reload view</button></div>;
    return this.props.children;
  }
}
