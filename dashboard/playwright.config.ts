import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  use: { baseURL: "http://127.0.0.1:5173", ...devices["Desktop Chrome"] },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --mode mock",
    port: 5173,
    reuseExistingServer: !process.env.CI,
    env: { VITE_USE_MOCKS: "true" },
  },
});
