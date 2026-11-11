import { defineConfig } from '@playwright/test'

// End-to-end tests drive the real UI against the real FastAPI backend and trained models.
// Start both servers first (scripts/start.ps1), then: npx playwright test
export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.TV_FRONTEND_URL ?? 'http://localhost:5173',
    viewport: { width: 1440, height: 900 },
    colorScheme: 'dark',
  },
})
