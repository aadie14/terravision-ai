// Types mirroring the FastAPI response contracts (backend/app/schemas.py).

export type Domain = 'vhr' | 'sentinel2'

export type AnalysisStatus = 'queued' | 'running' | 'completed' | 'failed'

export interface AnalysisSummary {
  id: string
  name: string
  status: AnalysisStatus
  source: string
  model_key: string
  threshold: number
  created_at: string
  started_at: string | null
  finished_at: string | null
  width: number | null
  height: number | null
  georeferenced: boolean
  has_ground_truth: boolean
  changed_fraction: number | null
  inference_seconds: number | null
  error_code: string | null
  error_message: string | null
}

export interface AnalysisDetail extends AnalysisSummary {
  inputs: Record<string, unknown> & { warnings?: string[] }
  artifacts: string[]
}

export interface AnalysisList {
  items: AnalysisSummary[]
  total: number
  limit: number
  offset: number
}

export interface Checkpoint {
  file: string
  sha256: string | null
  version: string
  run_name: string | null
  epoch: number | null
  val_f1: number | null
  val_iou: number | null
  trained_on: string | null
  saved_utc: string | null
  seed: number | null
  threshold?: number | null
}

export interface Metrics {
  tp: number
  fp: number
  tn: number
  fn: number
  precision: number | null
  recall: number | null
  f1: number | null
  iou: number | null
  accuracy: number | null
  undefined: string[]
  confusion_matrix: { labels: string[]; layout: string; matrix: number[][] }
  ground_truth_changed_fraction?: number
}

export interface Statistics {
  total_pixels: number
  changed_pixels: number
  changed_fraction: number
  changed_percent: number
  region_count: number
  largest_region_pixels: number
  mean_probability: number
  max_probability: number
  mean_probability_in_change: number | null
  uncertain_fraction: number
  probability_histogram: { bin_edges: number[]; counts: number[] }
}

export interface Bounds {
  south: number
  west: number
  north: number
  east: number
}

export interface Geospatial {
  crs: string
  crs_is_projected: boolean
  crs_units: string
  bounds: number[]
  bounds_wgs84: Bounds
  resolution: number[]
  region_count: number
  changed_area_m2: number
  changed_area_ha: number
  area_method: string
  regions_truncated: boolean
  footprint_area_m2: number
  footprint_area_km2: number
  map_overlay_bounds: Bounds
}

export interface ResultsPayload {
  analysis_id: string
  model: { key: string; display_name: string; family: string; parameters: number; checkpoint: Checkpoint }
  inference: { seconds: number; tiles: number; tile_size: number; tile_overlap: number; device: string; amp: boolean }
  image: { width: number; height: number }
  georeferenced: boolean
  has_ground_truth: boolean
  threshold: number
  statistics: Statistics
  metrics: Metrics | null
  geospatial: Geospatial | null
}

export interface AnalysisResults {
  analysis: AnalysisDetail
  results: ResultsPayload
  downloads: Record<string, string>
}

export interface Benchmark {
  dataset: string
  split: string
  num_patches?: number
  num_images?: number
  evaluation?: string
  threshold: number
  precision: number | null
  recall: number | null
  f1: number | null
  iou: number | null
  accuracy: number | null
  tp: number
  fp: number
  tn: number
  fn: number
  latency_ms_median: number
  latency_ms_p95: number
  throughput_patches_per_s: number | null
  peak_gpu_mb: number | null
  device: string
  gpu_name: string | null
  evaluated_utc: string
  confusion_matrix: Metrics['confusion_matrix']
}

export interface ModelInfo {
  key: string
  display_name: string
  family: 'transformer' | 'cnn'
  domain: Domain
  domain_label: string
  architecture: string
  reference: string
  tile_size: number
  checkpoint_file: string
  loaded: boolean
  available: boolean
  unavailable_reason: string | null
  parameters: number | null
  checkpoint: Checkpoint | null
  benchmark: Benchmark | null
}

export interface ModelsResponse {
  default: string
  models: ModelInfo[]
}

export interface Stats {
  total: number
  completed: number
  failed: number
  in_progress: number
  mean_changed_percent: number | null
  last_changed_percent: number | null
  mean_inference_seconds: number | null
  by_model: Record<string, number>
}

export interface SystemInfo {
  version: string
  time: string
  compute: {
    device: 'cuda' | 'cpu'
    requested: string
    cuda_available: boolean
    amp: boolean
    torch: string
    gpu_name?: string
    gpu_memory_total_mb?: number
    gpu_memory_free_mb?: number
    gpu_memory_allocated_mb?: number
    cuda_version?: string
  }
  platform: { os: string; python: string }
  database: { ok: boolean; engine: string }
  storage: { analyses_bytes: number; disk_free_bytes: number }
  models: { key: string; available: boolean; loaded: boolean }[]
  samples: number
}

export interface Health {
  status: 'ok' | 'degraded'
  version: string
  time: string
  database: boolean
  models_available: number
}

export interface AppConfig {
  upload: { max_file_mb: number; min_image_side: number; max_image_side: number; allowed_extensions: string[] }
  inference: { default_model: string; default_threshold: number; tile_overlap: number }
  satellite: { min_aoi_km: number; max_aoi_km: number }
  training_domain: string
}

export interface Sample {
  id: string
  name: string
  description: string
  dataset: string | null
  split: string | null
  has_ground_truth: boolean
  georeferenced: boolean
  resolution_note: string | null
  domain: Domain
  license: string | null
  attribution: string | null
  source_url: string | null
  t1_date: string | null
  t2_date: string | null
}

export interface Provider {
  key: string
  name: string
  description: string
  data_license: string
  documentation_url: string
  requires_credentials: boolean
  configured: boolean
  verified: boolean
}

export interface Scene {
  id: string
  provider: string
  collection: string
  acquired: string
  cloud_cover: number | null
  platform: string | null
  crs: string | null
  resolution_m: number
  aoi_coverage: number
  thumbnail_url: string | null
  extra: Record<string, unknown>
}

export interface SceneSearch {
  provider: Provider
  bbox: number[]
  count: number
  scenes: Scene[]
}

export interface GeocodeResult {
  name: string
  lat: number
  lon: number
  bbox: [number, number, number, number]
}

export interface EvalModelResult {
  model: string
  display_name: string
  precision: number | null
  recall: number | null
  f1: number | null
  iou: number | null
  accuracy: number | null
  tp: number
  fp: number
  tn: number
  fn: number
  mean_inference_seconds: number
  device: string
  per_sample: { sample_id: string; name: string; f1: number | null; iou: number | null }[]
  samples?: string[]
  dataset?: string | null
  threshold?: number
}

export interface SampleRun {
  id: string
  created_at: string
  scope: string
  samples: string[]
  threshold: number | null
  models: EvalModelResult[]
}

export interface CrossDomainResult {
  model: string
  display_name: string
  dataset: string
  trained_on: string
  precision: number | null
  recall: number | null
  f1: number | null
  iou: number | null
  num_images: number
}

export interface BenchmarkModel extends Benchmark {
  model: string
  display_name: string
  family: string
  architecture: string
  parameters: number
  checkpoint: string
}

export interface EvaluationSummary {
  benchmark: {
    generated_utc: string
    note: string
    datasets: Record<string, { split: string; evaluation: string | null; models: BenchmarkModel[] }>
    cross_domain: CrossDomainResult[]
    models: BenchmarkModel[]
  } | null
  benchmark_available: boolean
  figures: Record<string, string>
  sample_runs: SampleRun[]
}

export interface BBox {
  west: number
  south: number
  east: number
  north: number
}
