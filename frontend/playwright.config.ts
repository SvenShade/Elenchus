import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  webServer: {
    command:
      "uv run --extra web llm-mcts web --game examples/elenchus_v0_1.yaml --mock-llm --no-open --api-port 8897 --web-port 5173 --max-simulations 8",
    cwd: "..",
    url: "http://127.0.0.1:5173",
    timeout: 30_000,
    reuseExistingServer: false
  },
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "on-first-retry"
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 5"] } }
  ]
});
