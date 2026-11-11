import { Cloud, Crosshair, Eraser, Play, Search, SquareDashed } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { bboxSizeKm, GeoMap } from '../components/maps'
import { useToast } from '../components/Toasts'
import { Button, Card, cx, ErrorBanner, Field, inputClass, PageHeader, Spinner, Tag } from '../components/ui'
import { api, ApiError, artifactUrl } from '../lib/api'
import { dateTime, modelLabel } from '../lib/format'
import { loadPrefs, toApiError, useAsync } from '../lib/hooks'
import type { AnalysisSummary, BBox, Bounds, GeocodeResult, Scene } from '../lib/types'

interface Window_ {
  start: string
  end: string
}

export function MapPage() {
  const navigate = useNavigate()
  const { notify } = useToast()
  const base = useAsync(() => Promise.all([api.providers(), api.models(), api.config(), api.listAnalyses({ limit: 100, status: 'completed' })]), [])

  const [aoi, setAoi] = useState<BBox | null>(null)
  const [drawing, setDrawing] = useState(false)
  const [focus, setFocus] = useState<Bounds | BBox | null>(null)
  const [place, setPlace] = useState('')
  const [places, setPlaces] = useState<GeocodeResult[] | null>(null)
  const [geocoding, setGeocoding] = useState(false)
  const [provider, setProvider] = useState('earth_search')
  const [earlier, setEarlier] = useState<Window_>({ start: '2019-01-01', end: '2019-03-31' })
  const [later, setLater] = useState<Window_>({ start: '2025-01-01', end: '2025-03-31' })
  const [maxCloud, setMaxCloud] = useState(20)
  const [searching, setSearching] = useState(false)
  const [candidates, setCandidates] = useState<{ t1: Scene[]; t2: Scene[] } | null>(null)
  const [pick, setPick] = useState<{ t1: string | null; t2: string | null }>({ t1: null, t2: null })
  const [error, setError] = useState<ApiError | null>(null)
  const [running, setRunning] = useState(false)
  const [shown, setShown] = useState<{ id: string; bounds: Bounds } | null>(null)

  if (base.loading && !base.data) return <Spinner label="Loading map services…" />
  if (base.error || !base.data) return <ErrorBanner error={base.error ?? 'No data'} onRetry={base.reload} title="Map services unavailable" />
  const [providers, models, config, completed] = base.data
  const georeferenced = completed.items.filter((a) => a.georeferenced)
  const activeProvider = providers.providers.find((p) => p.key === provider)
  const usableModels = models.models.filter((m) => m.available)
  const s2Models = usableModels.filter((m) => m.domain === 'sentinel2')
  const preferredKey = loadPrefs().defaultModel ?? models.default
  const preferredFamily = usableModels.find((m) => m.key === preferredKey)?.family ?? 'transformer'
  const model = (s2Models.find((m) => m.family === preferredFamily) ?? s2Models[0] ?? usableModels[0])?.key ?? ''
  const size = aoi ? bboxSizeKm(aoi) : null
  const sizeProblem = size
    ? Math.max(size.width, size.height) > config.satellite.max_aoi_km
      ? `Too large — each side must be at most ${config.satellite.max_aoi_km} km.`
      : Math.min(size.width, size.height) < config.satellite.min_aoi_km
        ? `Too small — each side must be at least ${config.satellite.min_aoi_km} km.`
        : null
    : null

  const geocode = async () => {
    if (place.trim().length < 2) return
    setGeocoding(true)
    try {
      const { results } = await api.geocode(place.trim())
      setPlaces(results)
      if (results.length === 0) notify('info', `No place found for “${place.trim()}”.`)
    } catch (err) {
      notify('error', toApiError(err).message)
    } finally {
      setGeocoding(false)
    }
  }

  const search = async () => {
    if (!aoi) return
    setSearching(true)
    setError(null)
    setCandidates(null)
    setPick({ t1: null, t2: null })
    try {
      const common = { provider, bbox: aoi, max_cloud_cover: maxCloud, limit: 8 }
      const [a, b] = await Promise.all([
        api.searchScenes({ ...common, start: earlier.start, end: earlier.end }),
        api.searchScenes({ ...common, start: later.start, end: later.end }),
      ])
      setCandidates({ t1: a.scenes, t2: b.scenes })
      const full = (s: Scene[]) => s.find((x) => x.aoi_coverage >= 0.99)?.id ?? null
      setPick({ t1: full(a.scenes), t2: full(b.scenes) })
    } catch (err) {
      setError(toApiError(err))
    } finally {
      setSearching(false)
    }
  }

  const run = async () => {
    if (!aoi || !pick.t1 || !pick.t2 || !model) return
    setRunning(true)
    setError(null)
    try {
      const created = await api.retrieveScenes({
        provider, bbox: aoi, t1_scene_id: pick.t1, t2_scene_id: pick.t2, model,
      })
      notify('success', 'Imagery retrieved — running change detection.')
      navigate(`/results/${created.id}`)
    } catch (err) {
      const apiError = toApiError(err)
      setError(apiError)
      notify('error', apiError.message)
    } finally {
      setRunning(false)
    }
  }

  const showAnalysis = async (item: AnalysisSummary) => {
    if (shown?.id === item.id) {
      setShown(null)
      return
    }
    try {
      const res = await api.getResults(item.id)
      if (res.results.geospatial) {
        setShown({ id: item.id, bounds: res.results.geospatial.map_overlay_bounds })
        setFocus(res.results.geospatial.map_overlay_bounds)
      }
    } catch (err) {
      notify('error', toApiError(err).message)
    }
  }

  return (
    <>
      <PageHeader
        title="Interactive satellite map"
        description="Search a place, draw an area of interest, pick an earlier and a later Sentinel-2 acquisition and run change detection on the retrieved, georeferenced imagery."
      />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_400px]">
        <div className="space-y-4">
          <Card>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <form className="flex min-w-[220px] flex-1 gap-2" onSubmit={(e) => { e.preventDefault(); void geocode() }}>
                <input value={place} onChange={(e) => setPlace(e.target.value)} placeholder="Search a place, e.g. Yelahanka, Bengaluru"
                  aria-label="Search a place" className={inputClass} />
                <Button type="submit" loading={geocoding} disabled={place.trim().length < 2}><Search className="h-4 w-4" aria-hidden /> Search</Button>
              </form>
              <Button variant={drawing ? 'primary' : 'secondary'} onClick={() => setDrawing((d) => !d)} aria-pressed={drawing}>
                <SquareDashed className="h-4 w-4" aria-hidden /> {drawing ? 'Drag on the map…' : 'Draw area'}
              </Button>
              {aoi && <Button variant="ghost" onClick={() => { setAoi(null); setCandidates(null) }}><Eraser className="h-4 w-4" aria-hidden /> Clear</Button>}
            </div>
            {places && places.length > 0 && (
              <ul className="mb-3 divide-y divide-ink-700 rounded-lg border border-ink-700 text-sm">
                {places.map((p) => (
                  <li key={`${p.lat},${p.lon}`}>
                    <button type="button" className="flex w-full items-center gap-2 px-3 py-2 text-left text-ink-200 hover:bg-ink-800"
                      onClick={() => { setFocus({ west: p.bbox[0], south: p.bbox[1], east: p.bbox[2], north: p.bbox[3] }); setPlaces(null) }}>
                      <Crosshair className="h-3.5 w-3.5 shrink-0 text-accent-400" aria-hidden /> <span className="truncate">{p.name}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <GeoMap aoi={aoi} drawing={drawing} focus={focus}
              onDrawn={(bbox) => { setAoi(bbox); setDrawing(false); setCandidates(null) }}
              overlays={shown ? [
                { key: `t2-${shown.id}`, url: artifactUrl(shown.id, 'map_t2'), bounds: shown.bounds, opacity: 1 },
                { key: `mask-${shown.id}`, url: artifactUrl(shown.id, 'map_overlay'), bounds: shown.bounds, opacity: 0.7 },
              ] : []} />
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-400">
              <span>
                {aoi && size
                  ? `AOI ${size.width.toFixed(2)} × ${size.height.toFixed(2)} km · W ${aoi.west.toFixed(4)} S ${aoi.south.toFixed(4)} E ${aoi.east.toFixed(4)} N ${aoi.north.toFixed(4)}`
                  : 'No area selected — choose “Draw area” and drag a rectangle.'}
              </span>
              {sizeProblem && <span className="text-bad" role="alert">{sizeProblem}</span>}
            </div>
            <p className="mt-1 text-[11px] text-ink-500">Basemap tiles are for navigation only; analysis always uses retrieved, georeferenced scenes.</p>
          </Card>

          <Card title="Georeferenced results" subtitle="Completed analyses with spatial reference — toggle one to display it on the map">
            {georeferenced.length === 0 ? (
              <p className="text-sm text-ink-400">No georeferenced analysis yet. Run one from this page or from the Sentinel-2 bundled sample.</p>
            ) : (
              <ul className="space-y-1.5">
                {georeferenced.map((item) => (
                  <li key={item.id} className="flex items-center justify-between gap-3 rounded-lg border border-ink-700 px-3 py-2 text-sm">
                    <div className="min-w-0">
                      <div className="truncate text-ink-100">{item.name}</div>
                      <div className="text-xs text-ink-500">{modelLabel(item.model_key)} · {dateTime(item.created_at)}</div>
                    </div>
                    <Button variant={shown?.id === item.id ? 'primary' : 'secondary'} onClick={() => void showAnalysis(item)} aria-pressed={shown?.id === item.id}>
                      {shown?.id === item.id ? 'Hide' : 'Show on map'}
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>

        <div className="space-y-4">
          <Card title="Acquisition">
            <div className="space-y-4">
              <Field label="Imagery provider" htmlFor="provider" hint={activeProvider?.description}>
                <select id="provider" value={provider} onChange={(e) => { setProvider(e.target.value); setCandidates(null) }} className={inputClass}>
                  {providers.providers.map((p) => (
                    <option key={p.key} value={p.key}>{p.name}{p.requires_credentials && !p.configured ? ' — search only (no credentials)' : ''}</option>
                  ))}
                </select>
              </Field>
              {activeProvider && activeProvider.requires_credentials && !activeProvider.configured && (
                <p className="rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-ink-200">
                  Catalogue search works without an account, but image retrieval from this provider needs OAuth credentials
                  (TV_CDSE_CLIENT_ID / TV_CDSE_CLIENT_SECRET). Use Earth Search to run an analysis without an account.
                </p>
              )}
              <DateWindow label="Earlier period (T1)" value={earlier} onChange={setEarlier} />
              <DateWindow label="Later period (T2)" value={later} onChange={setLater} />
              <Field label={`Maximum cloud cover · ${maxCloud}%`} htmlFor="cloud">
                <input id="cloud" type="range" min={0} max={100} step={5} value={maxCloud} onChange={(e) => setMaxCloud(Number(e.target.value))} className="w-full accent-teal-400" />
              </Field>
              <Button variant="secondary" className="w-full" disabled={!aoi || Boolean(sizeProblem)} loading={searching} onClick={search}>
                <Search className="h-4 w-4" aria-hidden /> Find candidate scenes
              </Button>
            </div>
          </Card>

          {error && <ErrorBanner error={error} title="Satellite request failed" />}

          {candidates && (
            <Card title="Candidate scenes" subtitle="Sorted by AOI coverage, then cloud cover">
              <SceneList title="T1 · earlier" scenes={candidates.t1} selected={pick.t1} onSelect={(id) => setPick((p) => ({ ...p, t1: id }))} />
              <div className="my-4 border-t border-ink-700/60" />
              <SceneList title="T2 · later" scenes={candidates.t2} selected={pick.t2} onSelect={(id) => setPick((p) => ({ ...p, t2: id }))} />
              <Button variant="primary" className="mt-4 w-full py-2.5" loading={running}
                disabled={!pick.t1 || !pick.t2 || !model || Boolean(activeProvider?.requires_credentials && !activeProvider.configured)} onClick={run}>
                <Play className="h-4 w-4" aria-hidden /> Retrieve &amp; detect change
              </Button>
              <p className="mt-2 text-xs text-ink-500">
                Runs {model ? modelLabel(model) : 'no available model'}
                {s2Models.length ? ', fine-tuned on OSCD Sentinel-2 urban change (10 m).' : '. No Sentinel-2 model is available, so the LEVIR-CD model is used and the result is flagged as out-of-distribution.'}
              </p>
            </Card>
          )}
        </div>
      </div>
    </>
  )
}

function DateWindow({ label, value, onChange }: { label: string; value: Window_; onChange: (w: Window_) => void }) {
  const id = label.replace(/\W+/g, '-').toLowerCase()
  return (
    <fieldset>
      <legend className="mb-1.5 text-xs font-medium uppercase tracking-wider text-ink-400">{label}</legend>
      <div className="grid grid-cols-2 gap-2">
        <input type="date" aria-label={`${label} from`} id={`${id}-from`} value={value.start} max={value.end} min="2015-06-23"
          onChange={(e) => onChange({ ...value, start: e.target.value })} className={inputClass} />
        <input type="date" aria-label={`${label} to`} id={`${id}-to`} value={value.end} min={value.start}
          onChange={(e) => onChange({ ...value, end: e.target.value })} className={inputClass} />
      </div>
    </fieldset>
  )
}

function SceneList({ title, scenes, selected, onSelect }: { title: string; scenes: Scene[]; selected: string | null; onSelect: (id: string) => void }) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-400">{title}</h3>
      {scenes.length === 0 ? (
        <p className="text-sm text-ink-400">No scene matches this period and cloud limit. Widen the dates or raise the cloud threshold.</p>
      ) : (
        <ul className="max-h-64 space-y-1.5 overflow-y-auto pr-1" role="radiogroup" aria-label={title}>
          {scenes.map((scene) => {
            const partial = scene.aoi_coverage < 0.99
            return (
              <li key={scene.id}>
                <button type="button" role="radio" aria-checked={selected === scene.id} disabled={partial} onClick={() => onSelect(scene.id)}
                  className={cx('flex w-full items-center gap-3 rounded-lg border px-2.5 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50',
                    selected === scene.id ? 'border-accent-400 bg-accent-500/10' : 'border-ink-700 hover:border-ink-500')}>
                  {scene.thumbnail_url ? (
                    <img src={scene.thumbnail_url} alt="" loading="lazy" referrerPolicy="no-referrer" className="h-11 w-11 shrink-0 rounded object-cover" />
                  ) : <div className="h-11 w-11 shrink-0 rounded bg-ink-700" aria-hidden />}
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium text-ink-100">{scene.acquired.slice(0, 10)} <span className="text-xs font-normal text-ink-400">{scene.acquired.slice(11, 16)} UTC</span></div>
                    <div className="truncate font-mono text-[11px] text-ink-500">{scene.id}</div>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5">
                      <Tag><Cloud className="mr-1 inline h-3 w-3" aria-hidden />{scene.cloud_cover != null ? `${scene.cloud_cover.toFixed(1)}%` : 'n/a'}</Tag>
                      <Tag>{scene.platform ?? 'sentinel-2'}</Tag>
                      {scene.crs && <Tag>{scene.crs}</Tag>}
                      {partial && <Tag tone="warn">covers {Math.round(scene.aoi_coverage * 100)}% of AOI</Tag>}
                    </div>
                  </div>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
