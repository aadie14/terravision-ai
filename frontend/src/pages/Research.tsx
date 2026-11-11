import { ArrowDown, ExternalLink } from 'lucide-react'

import { Card, ErrorBanner, KeyValue, PageHeader, Spinner, Tag } from '../components/ui'
import references from '../data/references.json'
import { api } from '../lib/api'
import { dateTime, params, pct } from '../lib/format'
import { useAsync } from '../lib/hooks'
import type { ModelInfo } from '../lib/types'

const PIPELINE: [string, string][] = [
  ['Inputs T1, T2', 'Two co-registered RGB images, ImageNet-normalised, cut into 256×256 tiles'],
  ['Shared Swin-V2-T encoder', 'One set of weights encodes both dates → 4 feature scales (1/4, 1/8, 1/16, 1/32)'],
  ['Per-scale temporal fusion', 'concat(F1, F2, |F1 − F2|) → 1×1 conv → 3×3 conv at every scale'],
  ['FPN decoder', 'Top-down pathway with lateral additions, then multi-scale aggregation at 1/4 resolution'],
  ['Segmentation head', '3×3 conv → 1×1 conv → bilinear upsampling to full resolution: change logits'],
  ['Probability → mask', 'Sigmoid, overlap-blended across tiles, then thresholded (default 0.5)'],
]

export function ResearchPage() {
  const { data, error, loading, reload } = useAsync(() => Promise.all([api.models(), api.config()]), [])
  return (
    <>
      <PageHeader
        title="Research & model information"
        description="Vision Transformers for Satellite Land-Use Monitoring — how the models work, what they were trained on, and the literature the design builds on."
      />

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Primary model — Siamese Swin Transformer" subtitle="Data flow from image pair to change mask">
          <ol className="space-y-1.5">
            {PIPELINE.map(([step, detail], i) => (
              <li key={step}>
                <div className="rounded-lg border border-ink-700 bg-ink-900/60 px-3 py-2">
                  <div className="text-sm font-medium text-ink-100">{step}</div>
                  <div className="text-xs text-ink-400">{detail}</div>
                </div>
                {i < PIPELINE.length - 1 && <ArrowDown className="mx-auto my-0.5 h-4 w-4 text-ink-500" aria-hidden />}
              </li>
            ))}
          </ol>
        </Card>

        <div className="space-y-4">
          <Card title="Registered models" subtitle="Live from the backend model registry">
            {loading && !data ? <Spinner /> : error || !data ? <ErrorBanner error={error ?? 'No data'} onRetry={reload} /> : (
              <div className="space-y-5">
                {data[0].models.map((m) => <ModelBlock key={m.key} model={m} />)}
              </div>
            )}
          </Card>
          <Card title="Dataset and scope">
            <KeyValue rows={[
              ['Primary dataset', 'LEVIR-CD (Chen & Shi, 2020)'],
              ['Imagery', '637 pairs · 1024×1024 px · 0.5 m/pixel'],
              ['Official split', '445 train / 64 val / 128 test'],
              ['Tiling', '256×256 non-overlapping, inside each split'],
              ['Task', 'Binary building / urban change'],
              ['Licence', 'Academic use only'],
            ]} />
            <p className="mt-3 text-xs text-ink-500">
              Models are only validated for this domain{data ? ` (${data[1].training_domain})` : ''}. Other sensors, resolutions and change types
              (deforestation, disaster damage) are out-of-distribution until trained and evaluated on suitable datasets in Phase 2.
            </p>
          </Card>
        </div>
      </div>

      <Card className="mt-4" title="Literature" subtitle={`${references.length} peer-reviewed sources · bibliographic data verified against Crossref`}>
        <ul className="divide-y divide-ink-700/60">
          {references.map((ref) => (
            <li key={ref.id} className="py-4 first:pt-0 last:pb-0">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-sm font-medium text-ink-100">{ref.title}</p>
                  <p className="mt-0.5 text-xs text-ink-400">{ref.authors} ({ref.year}). {ref.venue}.</p>
                </div>
                <a href={`https://doi.org/${ref.doi}`} target="_blank" rel="noreferrer noopener" className="flex shrink-0 items-center gap-1 text-xs text-accent-300 hover:underline">
                  doi:{ref.doi} <ExternalLink className="h-3 w-3" aria-hidden />
                </a>
              </div>
              <div className="mt-2"><Tag>{ref.topic}</Tag></div>
              <dl className="mt-2 grid gap-x-6 gap-y-1.5 text-xs text-ink-300 md:grid-cols-2">
                <div><dt className="inline font-medium text-ink-200">Approach: </dt><dd className="inline">{ref.approach}</dd></div>
                <div><dt className="inline font-medium text-ink-200">Relevance: </dt><dd className="inline">{ref.relevance}</dd></div>
                <div><dt className="inline font-medium text-ok">Strength: </dt><dd className="inline">{ref.strength}</dd></div>
                <div><dt className="inline font-medium text-warn">Limitation: </dt><dd className="inline">{ref.limitation}</dd></div>
              </dl>
            </li>
          ))}
        </ul>
      </Card>
    </>
  )
}

function ModelBlock({ model }: { model: ModelInfo }) {
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold text-ink-100">{model.display_name}</h3>
        <Tag tone={model.family === 'transformer' ? 'accent' : 'default'}>{model.family}</Tag>
        <span className={`text-xs ${model.available ? 'text-ok' : 'text-bad'}`}>{model.available ? 'trained checkpoint loaded on demand' : 'no trained checkpoint'}</span>
      </div>
      <p className="mt-1 text-xs text-ink-400">{model.architecture}</p>
      <p className="mt-1 text-xs text-ink-500">Reference: {model.reference}</p>
      <div className="mt-2">
        <KeyValue rows={[
          ['Parameters', params(model.parameters)],
          ['Checkpoint', model.checkpoint ? <span key="c" className="font-mono text-xs">{model.checkpoint.version}</span> : 'n/a'],
          ['Best epoch / val F1', model.checkpoint ? `${model.checkpoint.epoch} / ${pct(model.checkpoint.val_f1)}` : 'n/a'],
          ['Saved', dateTime(model.checkpoint?.saved_utc)],
          ['Test F1 / IoU', model.benchmark ? `${pct(model.benchmark.f1)} / ${pct(model.benchmark.iou)}` : 'not benchmarked'],
        ]} />
      </div>
      {!model.available && model.unavailable_reason && <p className="mt-2 text-xs text-bad">{model.unavailable_reason}</p>}
    </div>
  )
}
