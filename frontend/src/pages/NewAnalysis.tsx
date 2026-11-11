import { Images, Map as MapIcon, MapPin, Play, Target, Upload } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import { FileDrop } from '../components/FileDrop'
import { ModelSelect } from '../components/ModelSelect'
import { useToast } from '../components/Toasts'
import { Button, Card, cx, EmptyState, ErrorBanner, Field, PageHeader, Spinner, Tag } from '../components/ui'
import { api, ApiError, sampleUrl } from '../lib/api'
import { DOMAIN_SHORT } from '../lib/format'
import { loadPrefs, toApiError, useAsync } from '../lib/hooks'
import type { Sample } from '../lib/types'

type Mode = 'upload' | 'sample'

export function NewAnalysisPage() {
  const navigate = useNavigate()
  const { notify } = useToast()
  const { data, error, loading, reload } = useAsync(() => Promise.all([api.models(), api.config(), api.samples()]), [])

  const [mode, setMode] = useState<Mode>('upload')
  const [t1, setT1] = useState<File | null>(null)
  const [t2, setT2] = useState<File | null>(null)
  const [gt, setGt] = useState<File | null>(null)
  const [sampleId, setSampleId] = useState<string | null>(null)
  const [model, setModel] = useState<string>('')
  const [autoModel, setAutoModel] = useState(true)
  const [threshold, setThreshold] = useState(0.5)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<ApiError | null>(null)

  useEffect(() => {
    if (!data) return
    const [models, config] = data
    const preferred = loadPrefs().defaultModel ?? models.default
    const usable = models.models.filter((m) => m.available)
    setModel((usable.find((m) => m.key === preferred) ?? usable[0])?.key ?? '')
    setThreshold(config.inference.default_threshold)
  }, [data])

  // Each checkpoint carries a threshold selected on its validation split; use it when the model changes.
  useEffect(() => {
    if (!data) return
    const ckpt = data[0].models.find((m) => m.key === model)?.checkpoint
    setThreshold(ckpt?.threshold ?? data[1].inference.default_threshold)
  }, [model, data])

  if (loading && !data) return <Spinner label="Loading models and samples…" />
  if (error || !data) return <ErrorBanner error={error ?? 'No data'} onRetry={reload} title="Cannot prepare a new analysis" />
  const [models, config, samples] = data
  const anyModel = models.models.some((m) => m.available)
  const auto = mode === 'upload' && autoModel
  const ready = (auto || Boolean(model)) && (mode === 'upload' ? Boolean(t1 && t2) : Boolean(sampleId))

  const submit = async () => {
    setSubmitting(true)
    setSubmitError(null)
    try {
      const created =
        mode === 'upload'
          ? await api.uploadAnalysis(auto ? { t1: t1!, t2: t2!, groundTruth: gt } : { t1: t1!, t2: t2!, groundTruth: gt, model, threshold })
          : await api.analyseSample(sampleId!, model, threshold)
      notify('success', 'Analysis queued — inference is starting.')
      navigate(`/results/${created.id}`)
    } catch (err) {
      const apiError = toApiError(err)
      setSubmitError(apiError)
      notify('error', apiError.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <>
      <PageHeader
        title="New analysis"
        description="Provide two co-registered images of the same place at different times. The model predicts, per pixel, where buildings changed between them."
      />

      <div className="mb-4 flex flex-wrap gap-2" role="tablist" aria-label="Input mode">
        {([
          ['upload', 'Upload images', Upload],
          ['sample', 'Bundled samples', Images],
        ] as const).map(([key, label, Icon]) => (
          <button key={key} type="button" role="tab" aria-selected={mode === key} onClick={() => setMode(key)}
            className={cx('flex items-center gap-2 rounded-lg border px-4 py-2 text-sm transition-colors',
              mode === key ? 'border-accent-400 bg-accent-500/10 text-accent-300' : 'border-ink-600 text-ink-300 hover:border-ink-500')}>
            <Icon className="h-4 w-4" aria-hidden /> {label}
          </button>
        ))}
        <Link to="/map" className="flex items-center gap-2 rounded-lg border border-ink-600 px-4 py-2 text-sm text-ink-300 hover:border-ink-500">
          <MapIcon className="h-4 w-4" aria-hidden /> Fetch from satellite map
        </Link>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-4">
          {mode === 'upload' ? (
            <Card title="Image pair" subtitle={`PNG, JPEG or GeoTIFF · ${config.upload.min_image_side}–${config.upload.max_image_side}px per side · up to ${config.upload.max_file_mb} MB each`}>
              <div className="grid gap-4 md:grid-cols-3">
                <FileDrop label="T1 · earlier image" file={t1} onChange={setT1} accept={config.upload.allowed_extensions} maxMb={config.upload.max_file_mb} />
                <FileDrop label="T2 · later image" file={t2} onChange={setT2} accept={config.upload.allowed_extensions} maxMb={config.upload.max_file_mb} />
                <FileDrop label="Ground-truth mask" optional hint="Binary change mask, same size as the images" file={gt} onChange={setGt}
                  accept={config.upload.allowed_extensions} maxMb={config.upload.max_file_mb} />
              </div>
              <ul className="mt-4 list-disc space-y-1 pl-5 text-xs text-ink-400">
                <li>Plain images must already be co-registered and have identical dimensions — mismatched pairs are rejected.</li>
                <li>GeoTIFFs keep their CRS; a T2 raster on a different grid is reprojected onto T1 when the footprints overlap ≥ 90%.</li>
                <li>Accuracy metrics (F1, IoU…) are only computed when you provide a ground-truth mask.</li>
              </ul>
            </Card>
          ) : (
            <Card title="Bundled samples" subtitle="Demonstration pairs shipped with the project — provenance and licence shown for each">
              {samples.samples.length === 0 ? (
                <EmptyState icon={<Images className="h-8 w-8" />} title="No samples installed">
                  Run <code className="font-mono text-ink-200">python scripts/build_samples.py</code> to create them.
                </EmptyState>
              ) : (
                <div className="grid gap-3 md:grid-cols-2">
                  {samples.samples.map((s) => (
                    <SampleCard key={s.id} sample={s} selected={sampleId === s.id} onSelect={() => {
                      setSampleId(s.id)
                      // pick the model trained for this imagery (same family as the current choice when possible)
                      const family = models.models.find((m) => m.key === model)?.family
                      const match = models.models
                        .filter((m) => m.available && m.domain === s.domain)
                        .sort((a, b) => Number(b.family === family) - Number(a.family === family))[0]
                      if (match) setModel(match.key)
                    }} />
                  ))}
                </div>
              )}
            </Card>
          )}
          {submitError && <ErrorBanner error={submitError} title="The analysis could not be started" />}
        </div>

        <div className="space-y-4">
          <Card title="Model">
            {anyModel ? (
              <>
                {mode === 'upload' && (
                  <label className="mb-3 flex cursor-pointer items-start gap-2 rounded-lg border border-ink-600 bg-ink-900/60 p-3 text-sm text-ink-200">
                    <input type="checkbox" checked={autoModel} onChange={(e) => setAutoModel(e.target.checked)} className="mt-0.5 accent-teal-400" />
                    <span>
                      Choose the model automatically from the imagery
                      <span className="mt-0.5 block text-xs text-ink-400">
                        GeoTIFFs with 10 m pixels use a Sentinel-2 model; other images use a LEVIR-CD model. The model&apos;s
                        validation-selected threshold is used.
                      </span>
                    </span>
                  </label>
                )}
                <div className={auto ? 'pointer-events-none opacity-40' : undefined} aria-disabled={auto}>
                  <ModelSelect models={models.models} value={model} onChange={setModel} />
                </div>
                <p className="mt-3 text-xs text-ink-500">
                  LEVIR-CD models expect 0.5 m aerial imagery; Sentinel-2 models expect 10 m Sentinel-2 true colour.
                  A mismatch is flagged on the result page.
                </p>
              </>
            ) : (
              <ErrorBanner error="No model has a trained checkpoint. Train one with `python -m ml.train` first." title="No model available" />
            )}
          </Card>
          <Card title="Decision threshold">
            {auto && <p className="mb-3 text-xs text-ink-400">Automatic mode uses the chosen model&apos;s own threshold; adjust it afterwards on the results page.</p>}
            <Field label={`Change probability ≥ ${threshold.toFixed(2)}`} htmlFor="threshold"
              hint="Pixels above this probability are marked as change. Defaults to the value selected on the model's validation split; it can be adjusted again on the results page.">
              <input id="threshold" type="range" min={0.05} max={0.95} step={0.05} value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))} className="w-full accent-teal-400" />
            </Field>
          </Card>
          <Button variant="primary" className="w-full py-3" disabled={!ready} loading={submitting} onClick={submit}>
            <Play className="h-4 w-4" aria-hidden /> Run change detection
          </Button>
          {!ready && (
            <p className="text-center text-xs text-ink-500">
              {mode === 'upload' ? 'Select a T1 and a T2 image to continue.' : 'Select a sample to continue.'}
            </p>
          )}
        </div>
      </div>
    </>
  )
}

function SampleCard({ sample, selected, onSelect }: { sample: Sample; selected: boolean; onSelect: () => void }) {
  return (
    <button type="button" onClick={onSelect} aria-pressed={selected}
      className={cx('rounded-xl border p-3 text-left transition-colors',
        selected ? 'border-accent-400 bg-accent-500/10' : 'border-ink-600 bg-ink-900/60 hover:border-ink-500')}>
      <div className="grid grid-cols-2 gap-1.5">
        <img src={sampleUrl(sample.id, 't1')} alt={`${sample.name} T1`} loading="lazy" className="aspect-square w-full rounded-md object-cover" />
        <img src={sampleUrl(sample.id, 't2')} alt={`${sample.name} T2`} loading="lazy" className="aspect-square w-full rounded-md object-cover" />
      </div>
      <div className="mt-2.5 text-sm font-medium text-ink-100">{sample.name}</div>
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        <Tag>sample data</Tag>
        {sample.has_ground_truth && <Tag tone="accent"><Target className="mr-1 inline h-3 w-3" aria-hidden />ground truth</Tag>}
        {sample.georeferenced && <Tag tone="accent"><MapPin className="mr-1 inline h-3 w-3" aria-hidden />georeferenced</Tag>}
        <Tag>{DOMAIN_SHORT[sample.domain] ?? sample.domain}</Tag>
      </div>
      <p className="mt-2 line-clamp-3 text-xs text-ink-400">{sample.description}</p>
      {sample.attribution && <p className="mt-1.5 line-clamp-2 text-[11px] text-ink-500">{sample.attribution}</p>}
    </button>
  )
}
