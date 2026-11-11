import { Download, Layers, MapPin, RefreshCw, Repeat } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { AnalysisTable } from '../components/AnalysisTable'
import { CompareSlider } from '../components/CompareSlider'
import { GeoMap } from '../components/maps'
import { useMaskOverlay } from '../components/maskOverlay'
import { useToast } from '../components/Toasts'
import {
  Button, Card, cx, EmptyState, ErrorBanner, Field, KeyValue, PageHeader, Spinner, StatCard, StatusBadge, WarningList,
} from '../components/ui'
import { api, artifactUrl, downloadUrl } from '../lib/api'
import { area, dateTime, modelLabel, num, params, pct, seconds } from '../lib/format'
import { loadPrefs, toApiError, useAsync } from '../lib/hooks'
import type { AnalysisDetail, AnalysisResults, Metrics, ModelInfo } from '../lib/types'

export function ResultsIndexPage() {
  const { data, error, loading, reload } = useAsync(() => api.listAnalyses({ limit: 25 }), [], 6000)
  return (
    <>
      <PageHeader title="Change detection results" description="Open an analysis to inspect its change mask, overlay, metrics and downloads." />
      {loading && !data ? <Spinner /> : error && !data ? <ErrorBanner error={error} onRetry={reload} /> : data && data.items.length ? (
        <Card><AnalysisTable items={data.items} /></Card>
      ) : (
        <EmptyState icon={<Layers className="h-8 w-8" />} title="No results yet" action={<Link to="/new"><Button variant="primary">New analysis</Button></Link>}>
          Results appear here as soon as an analysis has been started.
        </EmptyState>
      )}
    </>
  )
}

export function ResultsPage() {
  const { id = '' } = useParams()
  const [detail, setDetail] = useState<AnalysisDetail | null>(null)
  const [results, setResults] = useState<AnalysisResults | null>(null)
  const [error, setError] = useState<ReturnType<typeof toApiError> | null>(null)

  // Poll the analysis until it reaches a terminal state, then fetch the full results once.
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    setDetail(null)
    setResults(null)
    setError(null)
    const tick = async () => {
      try {
        const d = await api.getAnalysis(id)
        if (cancelled) return
        setDetail(d)
        if (d.status === 'completed') {
          const r = await api.getResults(id)
          if (!cancelled) setResults(r)
        } else if (d.status !== 'failed') {
          timer = setTimeout(tick, 1000)
        }
      } catch (err) {
        if (!cancelled) setError(toApiError(err))
      }
    }
    void tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [id])

  if (error) {
    return (
      <>
        <PageHeader title="Analysis" />
        <ErrorBanner error={error} title={error.status === 404 ? 'Analysis not found' : 'Could not load the analysis'} />
        <div className="mt-4"><Link to="/history" className="text-sm text-accent-300 hover:underline">Back to history</Link></div>
      </>
    )
  }
  if (!detail) return <Spinner label="Loading analysis…" />

  if (detail.status === 'failed') {
    return (
      <>
        <PageHeader title={detail.name || 'Analysis'} description={<StatusBadge status="failed" />} />
        <ErrorBanner error={detail.error_message ?? 'The analysis failed.'} title={`Analysis failed${detail.error_code ? ` (${detail.error_code})` : ''}`} />
        <WarningBlock detail={detail} />
      </>
    )
  }
  if (!results) {
    return (
      <>
        <PageHeader title={detail.name || 'Analysis'} description={<StatusBadge status={detail.status} />} />
        <Card>
          <div className="py-8 text-center">
            <p className="text-sm font-medium text-ink-100">
              {detail.status === 'queued' ? 'Waiting for the inference worker…' : `Running ${modelLabel(detail.model_key)} inference…`}
            </p>
            <p className="mt-1 text-xs text-ink-400">
              {detail.width}×{detail.height}px · sliding-window tiles · this page updates automatically
            </p>
            <div className="mx-auto mt-5 h-1.5 w-72 max-w-full overflow-hidden rounded-full bg-ink-700" role="progressbar" aria-label="Analysis in progress">
              <div className="progress-indeterminate h-full w-1/3 rounded-full bg-accent-400" />
            </div>
          </div>
        </Card>
        <WarningBlock detail={detail} />
      </>
    )
  }
  return <ResultsView key={id} initial={results} />
}

function WarningBlock({ detail }: { detail: AnalysisDetail }) {
  const warnings = detail.inputs.warnings ?? []
  return warnings.length ? <div className="mt-4"><WarningList warnings={warnings} /></div> : null
}

type View = 'slider' | 'side' | 'overlay' | 'mask' | 'confidence' | 'truth' | 'map'

function ResultsView({ initial }: { initial: AnalysisResults }) {
  const navigate = useNavigate()
  const { notify } = useToast()
  const [data, setData] = useState(initial)
  const [view, setView] = useState<View>('slider')
  const [opacity, setOpacity] = useState(() => loadPrefs().overlayOpacity)
  const [threshold, setThreshold] = useState(initial.results.threshold)
  const [version, setVersion] = useState(0)
  const [updating, setUpdating] = useState(false)
  const [showOverlay, setShowOverlay] = useState(true)
  const models = useAsync(() => api.models(), [])
  const debounce = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const { analysis, results } = data
  const id = analysis.id
  const has = (name: string) => analysis.artifacts.includes(name)
  const url = (name: string) => artifactUrl(id, name, version)
  const { overlay } = useMaskOverlay(url('mask'))

  const applyThreshold = (value: number) => {
    setThreshold(value)
    if (debounce.current) clearTimeout(debounce.current)
    debounce.current = setTimeout(async () => {
      setUpdating(true)
      try {
        const next = await api.setThreshold(id, value)
        setData(next)
        setVersion((v) => v + 1)
      } catch (err) {
        notify('error', toApiError(err).message)
        setThreshold(results.threshold)
      } finally {
        setUpdating(false)
      }
    }, 350)
  }
  useEffect(() => () => { if (debounce.current) clearTimeout(debounce.current) }, [])

  const views = useMemo(() => {
    const list: [View, string][] = [['slider', 'Before / after'], ['side', 'Side by side'], ['overlay', 'Overlay'], ['mask', 'Change mask'], ['confidence', 'Confidence']]
    if (results.has_ground_truth) list.push(['truth', 'Ground truth'])
    if (results.geospatial) list.push(['map', 'Map'])
    return list
  }, [results.has_ground_truth, results.geospatial])

  const stats = results.statistics
  const warnings = analysis.inputs.warnings ?? []

  return (
    <>
      <PageHeader
        title={analysis.name || 'Analysis'}
        description={
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <StatusBadge status={analysis.status} />
            <span className="font-mono text-xs">{id.slice(0, 8)}</span>
            <span>{dateTime(analysis.created_at)}</span>
            <span>{results.image.width}×{results.image.height}px</span>
            {results.georeferenced && <span className="flex items-center gap-1 text-accent-300"><MapPin className="h-3.5 w-3.5" aria-hidden />georeferenced</span>}
          </span>
        }
        actions={<RerunMenu analysisId={id} currentModel={results.model.key} models={models.data?.models ?? []}
          onStarted={(newId) => navigate(`/results/${newId}`)} />}
      />

      {warnings.length > 0 && <div className="mb-4"><WarningList warnings={warnings} /></div>}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Changed pixels" tone="accent" value={`${stats.changed_percent.toFixed(2)}%`}
          hint={`${num(stats.changed_pixels)} of ${num(stats.total_pixels)} px · ${num(stats.region_count)} regions`} />
        <StatCard label="Inference time" value={seconds(results.inference.seconds)}
          hint={`${results.inference.tiles} tiles · ${results.inference.device.toUpperCase()}${results.inference.amp ? ' · AMP' : ''}`} />
        {results.metrics ? (
          <>
            <StatCard label="F1-score" tone="ok" value={pct(results.metrics.f1)} hint="vs ground truth (change class)" />
            <StatCard label="IoU" tone="ok" value={pct(results.metrics.iou)} hint="intersection over union" />
          </>
        ) : (
          <>
            <StatCard label="Mean confidence" value={pct(stats.mean_probability_in_change, 1)} hint="mean probability inside predicted change" />
            <StatCard label="Uncertain pixels" value={pct(stats.uncertain_fraction, 2)} hint="probability between 0.30 and 0.70" />
          </>
        )}
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Card
          title="Visual comparison"
          action={updating ? <span className="flex items-center gap-1.5 text-xs text-ink-400"><RefreshCw className="h-3.5 w-3.5 animate-spin" aria-hidden />updating</span> : undefined}
        >
          <div className="mb-3 flex flex-wrap gap-1.5" role="tablist" aria-label="Result view">
            {views.map(([key, label]) => (
              <button key={key} type="button" role="tab" aria-selected={view === key} onClick={() => setView(key)}
                className={cx('rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
                  view === key ? 'bg-accent-500/15 text-accent-300' : 'text-ink-300 hover:bg-ink-700/70')}>
                {label}
              </button>
            ))}
          </div>

          {view === 'slider' && (
            <CompareSlider before={url('t1')} after={url('t2')} overlay={showOverlay && overlay ? { src: overlay, opacity } : null} />
          )}
          {view === 'side' && (
            <div className="grid gap-3 sm:grid-cols-2">
              <Figure src={url('t1')} caption="T1 · before" />
              <Figure src={url('t2')} caption="T2 · after" overlay={showOverlay && overlay ? { src: overlay, opacity } : null} />
            </div>
          )}
          {view === 'overlay' && <Figure src={url('t2')} caption="T2 with predicted change" overlay={overlay ? { src: overlay, opacity } : null} />}
          {view === 'mask' && <Figure src={url('mask')} caption={`Binary change mask · threshold ${results.threshold.toFixed(2)}`} pixelated />}
          {view === 'confidence' && (
            <>
              <Figure src={url('confidence')} caption="Change probability (model confidence)" />
              <div className="mt-2 flex items-center gap-2 text-xs text-ink-400">
                <span>0.0</span>
                <div className="h-2 flex-1 rounded-full" style={{ background: 'linear-gradient(90deg,#000004,#420a68,#932667,#dd513a,#fca50a,#fcffa4)' }} />
                <span>1.0</span>
              </div>
            </>
          )}
          {view === 'truth' && has('ground_truth') && (
            <div className="grid gap-3 sm:grid-cols-2">
              <Figure src={url('ground_truth')} caption="Ground-truth mask" pixelated />
              <div>
                <Figure src={url('error_map')} caption="Prediction vs ground truth" pixelated />
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-ink-300">
                  <Legend color="#2ecc71" label="true positive" />
                  <Legend color="#ff3b5c" label="false positive" />
                  <Legend color="#3498db" label="false negative" />
                </div>
              </div>
            </div>
          )}
          {view === 'map' && results.geospatial && (
            <>
              <GeoMap
                focus={results.geospatial.map_overlay_bounds}
                height={480}
                overlays={[
                  ...(has('map_t2') ? [{ key: `t2-${version}`, url: url('map_t2'), bounds: results.geospatial.map_overlay_bounds, opacity: 1 }] : []),
                  ...(showOverlay ? [{ key: `mask-${version}`, url: url('map_overlay'), bounds: results.geospatial.map_overlay_bounds, opacity }] : []),
                ]}
              />
              <p className="mt-2 text-xs text-ink-500">
                Imagery and mask are reprojected from {results.geospatial.crs} to Web Mercator for display. The basemap is for orientation only.
              </p>
            </>
          )}

          <div className="mt-4 grid gap-4 border-t border-ink-700/60 pt-4 sm:grid-cols-2">
            <Field label={`Threshold · ${threshold.toFixed(2)}`} htmlFor="res-threshold" hint="Re-thresholds the stored probability map; no new inference is run.">
              <input id="res-threshold" type="range" min={0.05} max={0.95} step={0.01} value={threshold}
                onChange={(e) => applyThreshold(Number(e.target.value))} className="w-full accent-teal-400" />
            </Field>
            <Field label={`Overlay opacity · ${Math.round(opacity * 100)}%`} htmlFor="res-opacity"
              hint={<label className="inline-flex cursor-pointer items-center gap-1.5"><input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} className="accent-teal-400" /> show change overlay</label>}>
              <input id="res-opacity" type="range" min={0} max={1} step={0.05} value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))} className="w-full accent-teal-400" />
            </Field>
          </div>
        </Card>

        <div className="space-y-4">
          <Card title="Model">
            <KeyValue rows={[
              ['Model', results.model.display_name],
              ['Family', results.model.family],
              ['Parameters', params(results.model.parameters)],
              ['Checkpoint', <span key="v" className="font-mono text-xs">{results.model.checkpoint.version}</span>],
              ['Trained on', results.model.checkpoint.trained_on ?? 'n/a'],
              ['Validation F1', pct(results.model.checkpoint.val_f1)],
              ['Tile / overlap', `${results.inference.tile_size}px / ${results.inference.tile_overlap}px`],
            ]} />
          </Card>

          {results.metrics ? <MetricsCard metrics={results.metrics} /> : (
            <Card title="Accuracy metrics">
              <p className="text-sm text-ink-400">
                No ground-truth mask was provided, so precision, recall, F1 and IoU cannot be computed. The figures above are
                descriptive statistics of the prediction only.
              </p>
            </Card>
          )}

          {results.geospatial && (
            <Card title="Geospatial">
              <KeyValue rows={[
                ['CRS', results.geospatial.crs],
                ['Pixel size', `${results.geospatial.resolution[0].toFixed(2)} × ${results.geospatial.resolution[1].toFixed(2)} ${results.geospatial.crs_units}`],
                ['Footprint', area(results.geospatial.footprint_area_m2)],
                ['Changed area', area(results.geospatial.changed_area_m2)],
                ['Regions', `${num(results.geospatial.region_count)}${results.geospatial.regions_truncated ? ' (truncated)' : ''}`],
              ]} />
              <p className="mt-3 text-xs text-ink-500">Area: {results.geospatial.area_method}.</p>
            </Card>
          )}
          {!results.geospatial && (
            <Card title="Geospatial">
              <p className="text-sm text-ink-400">
                The input images carry no spatial reference, so no map position, GeoTIFF export or real-world area is reported.
              </p>
            </Card>
          )}

          <Card title="Downloads">
            <div className="grid gap-2">
              {([
                ['mask', 'Change mask (PNG)'],
                ['overlay', 'Overlay (PNG)'],
                ['probability', 'Probability map (PNG)'],
                ['results', 'Results (JSON)'],
                ['mask_geotiff', 'Change mask (GeoTIFF)'],
                ['regions_geojson', 'Change regions (GeoJSON)'],
              ] as const).filter(([name]) => has(name)).map(([name, label]) => (
                <a key={name} href={downloadUrl(id, name)} download
                  className="flex items-center justify-between rounded-lg border border-ink-600 bg-ink-900/60 px-3 py-2 text-sm text-ink-100 hover:border-accent-500">
                  {label} <Download className="h-4 w-4 text-ink-400" aria-hidden />
                </a>
              ))}
            </div>
            {!results.georeferenced && <p className="mt-3 text-xs text-ink-500">GeoTIFF and GeoJSON exports require georeferenced input.</p>}
          </Card>
        </div>
      </div>
    </>
  )
}

function Figure({ src, caption, overlay, pixelated }: { src: string; caption: string; overlay?: { src: string; opacity: number } | null; pixelated?: boolean }) {
  return (
    <figure>
      <div className="relative overflow-hidden rounded-xl border border-ink-700 bg-ink-950">
        <img src={src} alt={caption} className={cx('block w-full', pixelated && 'pixelated')} />
        {overlay && <img src={overlay.src} alt="" aria-hidden className="pixelated pointer-events-none absolute inset-0 h-full w-full" style={{ opacity: overlay.opacity }} />}
      </div>
      <figcaption className="mt-1.5 text-xs text-ink-400">{caption}</figcaption>
    </figure>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-2.5 w-2.5 rounded-sm" style={{ background: color }} /> {label}
    </span>
  )
}

function MetricsCard({ metrics }: { metrics: Metrics }) {
  const [[tn, fp], [fn, tp]] = metrics.confusion_matrix.matrix
  return (
    <Card title="Accuracy vs ground truth" subtitle="Change class, pixel level">
      <KeyValue rows={[
        ['Precision', pct(metrics.precision)],
        ['Recall', pct(metrics.recall)],
        ['F1-score', pct(metrics.f1)],
        ['IoU', pct(metrics.iou)],
        ['Pixel accuracy', pct(metrics.accuracy)],
      ]} />
      {metrics.undefined.length > 0 && (
        <p className="mt-2 text-xs text-warn">
          {metrics.undefined.join(', ')} undefined for this pair (zero denominator — e.g. no change pixels in prediction and/or ground truth).
        </p>
      )}
      <table className="tabular mt-4 w-full border-separate border-spacing-1 text-center text-xs" aria-label="Confusion matrix">
        <thead>
          <tr><th /><th className="font-medium text-ink-400">Pred: no change</th><th className="font-medium text-ink-400">Pred: change</th></tr>
        </thead>
        <tbody>
          <tr><th className="pr-1 text-right font-medium text-ink-400">True: no change</th>
            <td className="rounded bg-ink-700/70 px-2 py-2">{num(tn)}<div className="text-[10px] text-ink-400">TN</div></td>
            <td className="rounded bg-bad/15 px-2 py-2">{num(fp)}<div className="text-[10px] text-ink-400">FP</div></td></tr>
          <tr><th className="pr-1 text-right font-medium text-ink-400">True: change</th>
            <td className="rounded bg-info/15 px-2 py-2">{num(fn)}<div className="text-[10px] text-ink-400">FN</div></td>
            <td className="rounded bg-ok/15 px-2 py-2">{num(tp)}<div className="text-[10px] text-ink-400">TP</div></td></tr>
        </tbody>
      </table>
    </Card>
  )
}

function RerunMenu({ analysisId, currentModel, models, onStarted }: {
  analysisId: string
  currentModel: string
  models: ModelInfo[]
  onStarted: (id: string) => void
}) {
  const { notify } = useToast()
  const [busy, setBusy] = useState<string | null>(null)
  const domain = models.find((m) => m.key === currentModel)?.domain
  const others = models.filter((m) => m.key !== currentModel && m.available && m.domain === domain)
  if (!others.length) return null
  return (
    <>
      {others.map((m) => (
        <Button key={m.key} loading={busy === m.key} onClick={async () => {
          setBusy(m.key)
          try {
            const created = await api.rerun(analysisId, m.key)
            notify('info', `Re-running with ${m.display_name}.`)
            onStarted(created.id)
          } catch (err) {
            notify('error', toApiError(err).message)
          } finally {
            setBusy(null)
          }
        }}>
          <Repeat className="h-4 w-4" aria-hidden /> Re-run with {m.display_name}
        </Button>
      ))}
    </>
  )
}
