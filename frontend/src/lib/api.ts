// Typed client for the TerraVision FastAPI backend. All calls are same-origin (Vite proxy).

import type {
  AnalysisDetail,
  AnalysisList,
  AnalysisResults,
  AnalysisStatus,
  AppConfig,
  BBox,
  EvaluationSummary,
  GeocodeResult,
  Health,
  ModelsResponse,
  Provider,
  Sample,
  SampleRun,
  SceneSearch,
  Stats,
  SystemInfo,
} from './types'

export class ApiError extends Error {
  readonly code: string
  readonly status: number
  readonly details: Record<string, unknown>

  constructor(code: string, message: string, status: number, details: Record<string, unknown> = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.details = details
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch {
    throw new ApiError('NETWORK_ERROR', 'Cannot reach the TerraVision backend. Is the API server running?', 0)
  }
  if (response.status === 204) return undefined as T
  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    body = null
  }
  if (!response.ok) {
    const err = (body as { error?: { code?: string; message?: string; details?: Record<string, unknown> } } | null)
      ?.error
    if (response.status >= 500 && !err) {
      throw new ApiError('BACKEND_UNAVAILABLE', 'The TerraVision backend is not responding.', response.status)
    }
    throw new ApiError(
      err?.code ?? 'HTTP_ERROR',
      err?.message ?? `Request failed with status ${response.status}`,
      response.status,
      err?.details ?? {},
    )
  }
  return body as T
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export interface UploadInput {
  t1: File
  t2: File
  groundTruth?: File | null
  /** Omit to let the server pick the model that matches the imagery (and that model's threshold). */
  model?: string
  threshold?: number
  name?: string
}

export const api = {
  health: () => request<Health>('/health'),
  system: () => request<SystemInfo>('/api/system'),
  config: () => request<AppConfig>('/api/config'),
  models: () => request<ModelsResponse>('/api/models'),
  samples: () => request<{ samples: Sample[] }>('/api/samples'),
  stats: () => request<Stats>('/api/analyses/stats'),

  listAnalyses: (params: { limit?: number; offset?: number; status?: AnalysisStatus } = {}) => {
    const query = new URLSearchParams()
    if (params.limit) query.set('limit', String(params.limit))
    if (params.offset) query.set('offset', String(params.offset))
    if (params.status) query.set('status', params.status)
    const qs = query.toString()
    return request<AnalysisList>(`/api/analyses${qs ? `?${qs}` : ''}`)
  },
  getAnalysis: (id: string) => request<AnalysisDetail>(`/api/analyses/${id}`),
  getResults: (id: string) => request<AnalysisResults>(`/api/analyses/${id}/results`),
  deleteAnalysis: (id: string) => request<void>(`/api/analyses/${id}`, { method: 'DELETE' }),
  setThreshold: (id: string, threshold: number) =>
    request<AnalysisResults>(`/api/analyses/${id}/threshold`, json({ threshold })),

  uploadAnalysis: (input: UploadInput) => {
    const form = new FormData()
    form.append('t1', input.t1)
    form.append('t2', input.t2)
    if (input.groundTruth) form.append('ground_truth', input.groundTruth)
    if (input.model) form.append('model', input.model)
    if (input.threshold !== undefined) form.append('threshold', String(input.threshold))
    if (input.name) form.append('name', input.name)
    return request<AnalysisDetail>('/api/analyses/upload', { method: 'POST', body: form })
  },
  analyseSample: (sampleId: string, model: string, threshold: number) =>
    request<AnalysisDetail>('/api/analyses', json({ sample_id: sampleId, model, threshold })),
  rerun: (analysisId: string, model: string, threshold?: number) =>
    request<AnalysisDetail>('/api/analyses', json({ rerun_of: analysisId, model, threshold })),

  evaluationSummary: () => request<EvaluationSummary>('/api/evaluation/summary'),
  runEvaluation: (threshold?: number) => request<SampleRun>('/api/evaluation', json({ threshold })),

  providers: () => request<{ providers: Provider[] }>('/api/satellite/providers'),
  geocode: (q: string) =>
    request<{ results: GeocodeResult[]; attribution: string }>(`/api/satellite/geocode?q=${encodeURIComponent(q)}`),
  searchScenes: (body: { provider: string; bbox: BBox; start: string; end: string; max_cloud_cover: number; limit: number }) =>
    request<SceneSearch>('/api/satellite/search', json(body)),
  retrieveScenes: (body: {
    provider: string
    bbox: BBox
    t1_scene_id: string
    t2_scene_id: string
    model: string
    threshold?: number
  }) => request<AnalysisDetail>('/api/satellite/retrieve', json(body)),
}

/** URL of an analysis artefact; `version` busts the cache after a re-threshold. */
export function artifactUrl(id: string, artifact: string, version?: string | number): string {
  return `/api/analyses/${id}/downloads/${artifact}${version !== undefined ? `?v=${version}` : ''}`
}

export function downloadUrl(id: string, artifact: string): string {
  return `/api/analyses/${id}/downloads/${artifact}?attachment=true`
}

export function sampleUrl(id: string, role: 't1' | 't2' | 'ground_truth'): string {
  return `/api/samples/${id}/${role}`
}
