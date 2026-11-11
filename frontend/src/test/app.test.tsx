import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { CompareSlider } from '../components/CompareSlider'
import { FileDrop, validateFile } from '../components/FileDrop'
import { ModelSelect } from '../components/ModelSelect'
import { bboxSizeKm } from '../components/maps'
import { ErrorBanner, StatusBadge } from '../components/ui'
import { ApiError, api, artifactUrl, downloadUrl } from '../lib/api'
import { area, bytes, params, pct, seconds } from '../lib/format'
import { fixtures, mockApi, renderApp } from './helpers'

const base = {
  'GET /health': { body: fixtures.health },
  'GET /api/system': { body: fixtures.system },
  'GET /api/models': { body: fixtures.models },
  'GET /api/config': { body: fixtures.config },
  'GET /api/samples': { body: fixtures.samples },
}

describe('formatting', () => {
  it('renders missing values as n/a instead of inventing numbers', () => {
    expect(pct(null)).toBe('n/a')
    expect(pct(0.91346)).toBe('91.35%')
    expect(seconds(0.135)).toBe('135 ms')
    expect(seconds(2.5)).toBe('2.50 s')
    expect(params(29976299)).toBe('29.98 M')
    expect(params(null)).toBe('n/a')
    expect(bytes(2048)).toBe('2.0 KB')
    expect(area(90000)).toBe('9.00 ha')
    expect(area(2_500_000)).toBe('2.50 km²')
  })
  it('builds artefact urls', () => {
    expect(artifactUrl('abc', 'mask', 3)).toBe('/api/analyses/abc/downloads/mask?v=3')
    expect(downloadUrl('abc', 'mask')).toBe('/api/analyses/abc/downloads/mask?attachment=true')
  })
  it('estimates AOI size', () => {
    const size = bboxSizeKm({ west: 77.6, south: 13.09, east: 77.66, north: 13.14 })
    expect(size.width).toBeGreaterThan(6)
    expect(size.height).toBeCloseTo(5.53, 1)
  })
})

describe('api client', () => {
  it('maps the backend error envelope to ApiError', async () => {
    mockApi({ 'GET /api/analyses/x': { status: 404, body: { error: { code: 'NOT_FOUND', message: 'Analysis x was not found', details: {} } } } })
    await expect(api.getAnalysis('x')).rejects.toMatchObject({ code: 'NOT_FOUND', status: 404, message: 'Analysis x was not found' })
  })
  it('reports an unreachable backend as NETWORK_ERROR', async () => {
    globalThis.fetch = (() => Promise.reject(new TypeError('failed'))) as typeof fetch
    await expect(api.health()).rejects.toBeInstanceOf(ApiError)
    await expect(api.health()).rejects.toMatchObject({ code: 'NETWORK_ERROR' })
  })
})

describe('components', () => {
  it('validates files before upload', () => {
    const accept = ['.png', '.tif']
    expect(validateFile(new File(['x'], 'a.png'), accept, 1)).toBeNull()
    expect(validateFile(new File(['x'], 'a.pdf'), accept, 1)).toMatch(/Unsupported file type/)
    expect(validateFile(new File([], 'a.png'), accept, 1)).toMatch(/empty/)
    expect(validateFile(new File([new Uint8Array(2 * 1024 * 1024)], 'a.tif'), accept, 1)).toMatch(/limit is 1 MB/)
  })
  it('FileDrop rejects an unsupported file and shows the reason', () => {
    const picked: (File | null)[] = []
    const { container } = render(<FileDrop label="T1" file={null} onChange={(f) => picked.push(f)} accept={['.png']} maxMb={1} />)
    const input = container.querySelector('input[type=file]') as HTMLInputElement
    fireEvent.change(input, { target: { files: [new File(['x'], 'notes.txt')] } })
    expect(screen.getByRole('alert')).toHaveTextContent('Unsupported file type')
    expect(picked).toEqual([null])
    fireEvent.change(input, { target: { files: [new File(['x'], 'ok.png')] } })
    expect(picked[1]?.name).toBe('ok.png')
  })
  it('ModelSelect disables models without a trained checkpoint', async () => {
    const chosen: string[] = []
    render(<ModelSelect models={fixtures.models.models as never} value="swin_siamese" onChange={(k) => chosen.push(k)} />)
    const radios = screen.getAllByRole('radio')
    expect(radios[0]).toHaveAttribute('aria-checked', 'true')
    expect(radios[1]).toBeDisabled()
    expect(screen.getByText('Checkpoint not found: cnn.pt')).toBeInTheDocument()
    await userEvent.click(radios[1])
    expect(chosen).toEqual([])
  })
  it('CompareSlider moves the reveal position', () => {
    render(<CompareSlider before="/a.png" after="/b.png" />)
    const slider = screen.getByLabelText('Before / after comparison position')
    fireEvent.change(slider, { target: { value: '20' } })
    expect(screen.getByAltText('T1 · before')).toHaveStyle({ clipPath: 'inset(0 80% 0 0)' })
  })
  it('ErrorBanner shows message, code and retry', async () => {
    let retried = 0
    render(<ErrorBanner error={new ApiError('MODEL_UNAVAILABLE', 'No checkpoint', 503)} onRetry={() => { retried += 1 }} />)
    expect(screen.getByRole('alert')).toHaveTextContent('No checkpoint')
    expect(screen.getByText('MODEL_UNAVAILABLE')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /retry/i }))
    expect(retried).toBe(1)
    render(<StatusBadge status="failed" />)
    expect(screen.getByText('failed')).toBeInTheDocument()
  })
})

describe('navigation and pages', () => {
  it('shows live statistics on the overview and navigates through the sidebar', async () => {
    mockApi({
      ...base,
      'GET /api/analyses/stats': { body: fixtures.stats },
      'GET /api/analyses': { body: { items: [fixtures.analysis()], total: 1, limit: 8, offset: 0 } },
      'GET /api/evaluation/summary': { body: { benchmark: null, benchmark_available: false, figures: {}, sample_runs: [] } },
    })
    renderApp('/')
    expect(await screen.findByTestId('stat-Total analyses')).toHaveTextContent('3')
    expect(screen.getByTestId('stat-Failed')).toHaveTextContent('1')
    expect(screen.getByTestId('stat-Changed area')).toHaveTextContent('7.25%')
    expect(screen.getAllByText('Test GPU').length).toBeGreaterThan(0)
    expect(screen.getByText('Demo pair')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('link', { name: 'Model Comparison' }))
    expect(await screen.findByText('No benchmark has been run yet')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('link', { name: 'Analysis History' }))
    expect(await screen.findByRole('heading', { name: 'Analysis history' })).toBeInTheDocument()
  })

  it('shows empty states when nothing has been run', async () => {
    mockApi({ ...base, 'GET /api/analyses/stats': { body: fixtures.emptyStats }, 'GET /api/analyses': { body: { items: [], total: 0, limit: 8, offset: 0 } } })
    renderApp('/')
    expect(await screen.findByText('No analyses yet')).toBeInTheDocument()
    expect(screen.getByTestId('stat-Changed area')).toHaveTextContent('n/a')
  })

  it('shows an error state with retry when the backend is down', async () => {
    mockApi({ 'GET /': { status: 500, body: null } })
    renderApp('/')
    expect(await screen.findByText('Dashboard unavailable')).toBeInTheDocument()
    expect(screen.getByText('Backend offline')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
  })

  it('renders a not-found page for unknown routes', async () => {
    mockApi(base)
    renderApp('/nowhere')
    expect(await screen.findByText('Page not found')).toBeInTheDocument()
  })
})

describe('upload workflow', () => {
  const id = 'a'.repeat(32)
  const routes = {
    ...base,
    [`GET /api/analyses/${id}/results`]: { body: { analysis: fixtures.analysis(), results: fixtures.results(), downloads: {} } },
    [`GET /api/analyses/${id}`]: { body: fixtures.analysis() },
  }

  it('uploads a pair, then shows results without accuracy metrics when no ground truth exists', async () => {
    const { calls } = mockApi({ ...routes, 'POST /api/analyses/upload': { status: 202, body: fixtures.analysis({ status: 'queued' }) } })
    const { container } = renderApp('/new')
    const run = await screen.findByRole('button', { name: /run change detection/i })
    expect(run).toBeDisabled()
    const inputs = container.querySelectorAll('input[type=file]')
    fireEvent.change(inputs[0], { target: { files: [new File(['a'], 'before.png', { type: 'image/png' })] } })
    fireEvent.change(inputs[1], { target: { files: [new File(['b'], 'after.png', { type: 'image/png' })] } })
    await waitFor(() => expect(run).toBeEnabled())
    await userEvent.click(run)

    expect(await screen.findByRole('heading', { name: 'Demo pair' })).toBeInTheDocument()
    const post = calls.find((c) => c.method === 'POST')!
    const form = post.init!.body as FormData
    expect((form.get('t1') as File).name).toBe('before.png')
    expect((form.get('t2') as File).name).toBe('after.png')
    expect(form.get('model')).toBeNull() // automatic: the server matches the model to the imagery
    expect(form.get('threshold')).toBeNull()
    expect(form.get('ground_truth')).toBeNull()

    expect(screen.getByTestId('stat-Changed pixels')).toHaveTextContent('9.50%')
    expect(screen.getByText(/No ground-truth mask was provided/)).toBeInTheDocument()
    expect(screen.queryByTestId('stat-F1-score')).not.toBeInTheDocument()
    expect(screen.getByText(/carry no spatial reference/)).toBeInTheDocument()
    expect(screen.queryByText('Change mask (GeoTIFF)')).not.toBeInTheDocument()
    expect(screen.getByText('Change mask (PNG)').closest('a')).toHaveAttribute('href', `/api/analyses/${id}/downloads/mask?attachment=true`)
  })

  it('sends the chosen model and threshold when automatic selection is turned off', async () => {
    const { calls } = mockApi({ ...routes, 'POST /api/analyses/upload': { status: 202, body: fixtures.analysis({ status: 'queued' }) } })
    const { container } = renderApp('/new')
    await userEvent.click(await screen.findByRole('checkbox', { name: /choose the model automatically/i }))
    const inputs = container.querySelectorAll('input[type=file]')
    fireEvent.change(inputs[0], { target: { files: [new File(['a'], 'a.png')] } })
    fireEvent.change(inputs[1], { target: { files: [new File(['b'], 'b.png')] } })
    const run = screen.getByRole('button', { name: /run change detection/i })
    await waitFor(() => expect(run).toBeEnabled())
    await userEvent.click(run)
    await screen.findByRole('heading', { name: 'Demo pair' })
    const form = calls.find((c) => c.method === 'POST')!.init!.body as FormData
    expect(form.get('model')).toBe('swin_siamese')
    expect(form.get('threshold')).toBe('0.5')
  })

  it('shows the server validation error when the upload is rejected', async () => {
    mockApi({ ...routes, 'POST /api/analyses/upload': { status: 422, body: { error: { code: 'IMAGE_DIMENSION_MISMATCH', message: 'T1 is 96x96px but T2 is 128x96px.', details: {} } } } })
    const { container } = renderApp('/new')
    const run = await screen.findByRole('button', { name: /run change detection/i })
    const inputs = container.querySelectorAll('input[type=file]')
    fireEvent.change(inputs[0], { target: { files: [new File(['a'], 'a.png')] } })
    fireEvent.change(inputs[1], { target: { files: [new File(['b'], 'b.png')] } })
    await waitFor(() => expect(run).toBeEnabled())
    await userEvent.click(run)
    expect(await screen.findByText('The analysis could not be started')).toBeInTheDocument()
    expect(screen.getAllByText('T1 is 96x96px but T2 is 128x96px.').length).toBeGreaterThan(0)
    expect(screen.getByText('IMAGE_DIMENSION_MISMATCH')).toBeInTheDocument()
  })

  it('starts an analysis from a bundled sample', async () => {
    const { calls } = mockApi({ ...routes, 'POST /api/analyses': { status: 202, body: fixtures.analysis() } })
    renderApp('/new')
    await userEvent.click(await screen.findByRole('tab', { name: /bundled samples/i }))
    await userEvent.click(screen.getByRole('button', { name: /LEVIR sample/ }))
    await userEvent.click(screen.getByRole('button', { name: /run change detection/i }))
    await screen.findByRole('heading', { name: 'Demo pair' })
    const post = calls.find((c) => c.method === 'POST')!
    expect(JSON.parse(post.init!.body as string)).toEqual({ sample_id: 'levir-1', model: 'swin_siamese', threshold: 0.5 })
  })
})

describe('results page states', () => {
  const id = 'a'.repeat(32)
  it('shows metrics and the confusion matrix when ground truth exists', async () => {
    const metrics = { tp: 90, fp: 10, tn: 880, fn: 20, precision: 0.9, recall: 0.8182, f1: 0.8571, iou: 0.75, accuracy: 0.97, undefined: [], confusion_matrix: { labels: [], layout: '', matrix: [[880, 10], [20, 90]] } }
    mockApi({
      ...base,
      [`GET /api/analyses/${id}/results`]: { body: { analysis: fixtures.analysis({ has_ground_truth: true }), results: fixtures.results({ has_ground_truth: true, metrics }), downloads: {} } },
      [`GET /api/analyses/${id}`]: { body: fixtures.analysis({ has_ground_truth: true }) },
    })
    renderApp(`/results/${id}`)
    expect(await screen.findByTestId('stat-F1-score')).toHaveTextContent('85.71%')
    expect(screen.getByTestId('stat-IoU')).toHaveTextContent('75.00%')
    const matrix = screen.getByRole('table', { name: 'Confusion matrix' })
    expect(within(matrix).getByText('880')).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Ground truth' })).toBeInTheDocument()
  })

  it('shows the failure reason for a failed analysis', async () => {
    mockApi({ ...base, [`GET /api/analyses/${id}`]: { body: fixtures.analysis({ status: 'failed', error_code: 'INFERENCE_FAILED', error_message: 'CUDA out of memory' }) } })
    renderApp(`/results/${id}`)
    expect(await screen.findByText('Analysis failed (INFERENCE_FAILED)')).toBeInTheDocument()
    expect(screen.getByText('CUDA out of memory')).toBeInTheDocument()
  })

  it('shows progress while the analysis is running', async () => {
    mockApi({ ...base, [`GET /api/analyses/${id}`]: { body: fixtures.analysis({ status: 'running' }) } })
    renderApp(`/results/${id}`)
    expect(await screen.findByRole('progressbar', { name: 'Analysis in progress' })).toBeInTheDocument()
  })

  it('shows not-found for an unknown analysis', async () => {
    mockApi({ ...base, [`GET /api/analyses/${id}`]: { status: 404, body: { error: { code: 'NOT_FOUND', message: 'Analysis was not found', details: {} } } } })
    renderApp(`/results/${id}`)
    expect(await screen.findByText('Analysis not found')).toBeInTheDocument()
  })
})
