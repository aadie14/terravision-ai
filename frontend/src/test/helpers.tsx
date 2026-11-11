import { render } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'

import App from '../App'
import { ToastProvider } from '../components/Toasts'

type Route = { status?: number; body: unknown } | ((init?: RequestInit) => { status?: number; body: unknown })

/** Install a fetch mock keyed by "METHOD path-prefix"; unmatched requests fail loudly. */
export function mockApi(routes: Record<string, Route>) {
  const calls: { method: string; url: string; init?: RequestInit }[] = []
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const method = (init?.method ?? 'GET').toUpperCase()
    calls.push({ method, url, init })
    const key = Object.keys(routes)
      .filter((k) => {
        const [m, p] = k.split(' ')
        return m === method && url.split('?')[0] === p
      })[0] ?? Object.keys(routes).filter((k) => {
        const [m, p] = k.split(' ')
        return m === method && url.startsWith(p)
      }).sort((a, b) => b.length - a.length)[0]
    if (!key) throw new Error(`Unmocked request: ${method} ${url}`)
    const route = routes[key]
    const { status = 200, body } = typeof route === 'function' ? route(init) : route
    return new Response(status === 204 ? null : JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
  })
  vi.stubGlobal('fetch', fn)
  return { calls, fn }
}

export function renderApp(path = '/') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
}

export function renderWithProviders(ui: ReactElement, path = '/') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <ToastProvider>{ui}</ToastProvider>
    </MemoryRouter>,
  )
}

const checkpoint = { file: 'x.pt', sha256: 'abc', version: 'run@epoch5#abc', run_name: 'run', epoch: 5, val_f1: 0.9, val_iou: 0.8, trained_on: 'LEVIR-CD', saved_utc: '2026-11-11T10:00:00+00:00', seed: 42 }

export const fixtures = {
  health: { status: 'ok', version: '0.1.0', time: '2026-11-11T10:00:00Z', database: true, models_available: 1 },
  system: {
    version: '0.1.0', time: '2026-11-11T10:00:00Z',
    compute: { device: 'cuda', requested: 'auto', cuda_available: true, amp: true, torch: '2.11.0', gpu_name: 'Test GPU', gpu_memory_total_mb: 8000, gpu_memory_free_mb: 6000, gpu_memory_allocated_mb: 100, cuda_version: '12.8' },
    platform: { os: 'Windows 11', python: '3.12.10' }, database: { ok: true, engine: 'sqlite' },
    storage: { analyses_bytes: 2048, disk_free_bytes: 5e10 }, models: [], samples: 1,
  },
  config: {
    upload: { max_file_mb: 50, min_image_side: 32, max_image_side: 4096, allowed_extensions: ['.jpeg', '.jpg', '.png', '.tif', '.tiff'] },
    inference: { default_model: 'swin_siamese', default_threshold: 0.5, tile_overlap: 64 },
    satellite: { min_aoi_km: 1, max_aoi_km: 25 }, training_domain: 'LEVIR-CD',
  },
  models: {
    default: 'swin_siamese',
    models: [
      { key: 'swin_siamese', display_name: 'Siamese Swin Transformer', family: 'transformer', domain: 'vhr', domain_label: 'LEVIR', architecture: 'Swin encoder', reference: 'ref', tile_size: 256, checkpoint_file: 'swin.pt', loaded: false, available: true, unavailable_reason: null, parameters: 29976299, checkpoint, benchmark: null },
      { key: 'fc_siam_diff', display_name: 'FC-Siam-diff (CNN baseline)', family: 'cnn', domain: 'vhr', domain_label: 'LEVIR', architecture: 'U-Net', reference: 'ref', tile_size: 256, checkpoint_file: 'cnn.pt', loaded: false, available: false, unavailable_reason: 'Checkpoint not found: cnn.pt', parameters: null, checkpoint: null, benchmark: null },
    ],
  },
  samples: { samples: [{ id: 'levir-1', name: 'LEVIR sample', description: 'd', dataset: 'LEVIR-CD', split: 'test', has_ground_truth: true, georeferenced: false, resolution_note: null, domain: 'vhr', license: 'academic', attribution: 'Chen & Shi', source_url: null, t1_date: null, t2_date: null }] },
  stats: { total: 3, completed: 2, failed: 1, in_progress: 0, mean_changed_percent: 7.25, last_changed_percent: 9.5, mean_inference_seconds: 0.42, by_model: { swin_siamese: 3 } },
  emptyStats: { total: 0, completed: 0, failed: 0, in_progress: 0, mean_changed_percent: null, last_changed_percent: null, mean_inference_seconds: null, by_model: {} },
  analysis: (over: Record<string, unknown> = {}) => ({
    id: 'a'.repeat(32), name: 'Demo pair', status: 'completed', source: 'upload', model_key: 'swin_siamese', threshold: 0.5,
    created_at: '2026-11-11T10:00:00Z', started_at: null, finished_at: null, width: 256, height: 256, georeferenced: false,
    has_ground_truth: false, changed_fraction: 0.095, inference_seconds: 0.31, error_code: null, error_message: null,
    inputs: { warnings: [] }, artifacts: ['t1', 't2', 'mask', 'overlay', 'probability', 'confidence', 'results'], ...over,
  }),
  results: (over: Record<string, unknown> = {}) => ({
    analysis_id: 'a'.repeat(32),
    model: { key: 'swin_siamese', display_name: 'Siamese Swin Transformer', family: 'transformer', parameters: 29976299, checkpoint },
    inference: { seconds: 0.31, tiles: 4, tile_size: 256, tile_overlap: 64, device: 'cuda', amp: true },
    image: { width: 256, height: 256 }, georeferenced: false, has_ground_truth: false, threshold: 0.5,
    statistics: { total_pixels: 65536, changed_pixels: 6226, changed_fraction: 0.095, changed_percent: 9.5, region_count: 3, largest_region_pixels: 4000, mean_probability: 0.1, max_probability: 0.99, mean_probability_in_change: 0.93, uncertain_fraction: 0.01, probability_histogram: { bin_edges: [], counts: [] } },
    metrics: null, geospatial: null, ...over,
  }),
}
