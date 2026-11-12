import { create } from "zustand";

type Theme = "dark" | "light";
interface DashboardState {
  streamId: string;
  range: number;
  theme: Theme;
  setStreamId: (streamId: string) => void;
  setRange: (range: number) => void;
  toggleTheme: () => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  streamId: "demo-store",
  range: 24,
  theme: (localStorage.getItem("rv-theme") as Theme | null) ?? "dark",
  setStreamId: (streamId) => set({ streamId }),
  setRange: (range) => set({ range }),
  toggleTheme: () =>
    set((state) => {
      const theme = state.theme === "dark" ? "light" : "dark";
      localStorage.setItem("rv-theme", theme);
      return { theme };
    }),
}));

export function rangeBounds(hours: number): [string, string] {
  const end = new Date();
  const start = new Date(end.getTime() - hours * 60 * 60 * 1000);
  return [start.toISOString(), end.toISOString()];
}
