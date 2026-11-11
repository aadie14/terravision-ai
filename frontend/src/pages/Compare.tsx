import { BarChart3, FlaskConical } from 'lucide-react'
import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { useToast } from '../components/Toasts'
import { Button, Card, EmptyState, ErrorBanner, PageHeader, Spinner, Tag } from '../components/ui'
import { api } from '../lib/api'
import { dateTime, MODEL_COLORS, num, params, pct } from '../lib/format'
import { toApiError, useAsync } from '../lib/hooks'
import type { BenchmarkModel, CrossDomainResult, SampleRun } from '../lib/types'

const METRICS = ['precision', 'recall', 'f1', 'iou', 'accuracy'] as const
const LABELS: Record<(typeof METRICS)[number], string> = { precision: 'Precision', recall: 'Recall', f1: 'F1', iou: 'IoU', accuracy: 'Accuracy' }

export function ComparePage() {
  const { notify } = useToast()
  const { data, error, loading, reload } = useAsync(() => api.evaluationSummary(), [])
  const [running, setRunning] = useState(false)
  const [latest, setLatest] = useState<SampleRun | null>(null)

  if (loading && !data) return <Spinner label="Loading evaluation results…" />
  if (error || !data) return <ErrorBanner error={error ?? 'No data'} onRetry={reload} title="Evaluation results unavailable" />

  const bench = data.benchmark_available ? data.benchmark : null
  const sampleRun = latest ?? data.sample_runs[0] ?? null

  const runSamples = async () => {
    setRunning(true)
    try {
      setLatest(await api.runEvaluation())
      notify('success', 'In-app evaluation finished.')
    } catch (err) {
      notify('error', toApiError(err).message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <>
      <PageHeader
        title="Model comparison"
        description="Transformer versus CNN on identical data. Every number on this page is read from evaluation runs executed on this machine — nothing is hard-coded."
      />

      {bench ? (
        <div className="space-y-8">
          {Object.entries(bench.datasets).map(([name, ds]) => (
            <section key={name} aria-label={`${name} benchmark`}>
              <h2 className="mb-3 text-lg font-semibold text-ink-100">{DATASET_TITLES[name] ?? name}</h2>
              <BenchmarkSection dataset={name} evaluation={ds.evaluation} models={ds.models} generated={bench.generated_utc} figures={data.figures} />
            </section>
          ))}
          {bench.cross_domain.length > 0 && <CrossDomainCard rows={bench.cross_domain} oscd={bench.datasets.OSCD?.models ?? []} />}
        </div>
      ) : (
        <EmptyState icon={<BarChart3 className="h-8 w-8" />} title="No benchmark has been run yet">
          Train the models, then run <code className="font-mono text-ink-200">python -m ml.evaluate --cross-domain</code> to evaluate them on the
          held-out LEVIR-CD and OSCD test data. Results will appear here automatically.
        </EmptyState>
      )}

      <Card className="mt-4" title="In-app check on bundled samples"
        subtitle="Runs every available model over the labelled sample pairs of its own imagery domain (a quick live demonstration, not the benchmark)"
        action={<Button variant="primary" loading={running} onClick={runSamples}><FlaskConical className="h-4 w-4" aria-hidden /> Run now</Button>}>
        {sampleRun ? <SampleRunTable run={sampleRun} /> : (
          <p className="text-sm text-ink-400">No in-app evaluation has been run yet. Press “Run now” to evaluate the models live.</p>
        )}
      </Card>
    </>
  )
}

const DATASET_TITLES: Record<string, string> = {
  'LEVIR-CD': 'LEVIR-CD — 0.5 m very-high-resolution building change',
  OSCD: 'OSCD — real Sentinel-2 (10 m) urban change',
}

function CrossDomainCard({ rows, oscd }: { rows: CrossDomainResult[]; oscd: BenchmarkModel[] }) {
  return (
    <Card title="Domain shift: does a LEVIR-CD model work on Sentinel-2?"
      subtitle="Same OSCD test cities, models trained only on 0.5 m LEVIR-CD imagery versus models fine-tuned on OSCD">
      <div className="overflow-x-auto">
        <table className="tabular w-full min-w-[620px] text-left text-sm">
          <thead className="text-xs uppercase tracking-wider text-ink-400">
            <tr className="border-b border-ink-700/70">
              <th className="px-3 py-2 font-medium">Model</th><th className="px-3 py-2 font-medium">Trained on</th>
              <th className="px-3 py-2 text-right font-medium">Precision</th><th className="px-3 py-2 text-right font-medium">Recall</th>
              <th className="px-3 py-2 text-right font-medium">F1</th><th className="px-3 py-2 text-right font-medium">IoU</th>
            </tr>
          </thead>
          <tbody>
            {[...rows.map((r) => ({ ...r, trained: 'LEVIR-CD only' })), ...oscd.map((m) => ({ ...m, trained: 'LEVIR-CD → OSCD fine-tune' }))].map((r) => (
              <tr key={`${r.model}-${r.trained}`} className="border-b border-ink-800 last:border-0">
                <td className="px-3 py-2.5 font-medium text-ink-100">{r.display_name}</td>
                <td className="px-3 py-2.5 text-ink-300">{r.trained}</td>
                <td className="px-3 py-2.5 text-right">{pct(r.precision)}</td>
                <td className="px-3 py-2.5 text-right">{pct(r.recall)}</td>
                <td className="px-3 py-2.5 text-right">{pct(r.f1)}</td>
                <td className="px-3 py-2.5 text-right">{pct(r.iou)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-ink-500">
        This is why a LEVIR-CD model reports almost no change on Sentinel-2 imagery: at 10 m a building is one or two pixels. The Sentinel-2 models
        are the ones the satellite map uses.
      </p>
    </Card>
  )
}

function BenchmarkSection({ dataset, evaluation, models, generated, figures }: { dataset: string; evaluation: string | null; models: BenchmarkModel[]; generated: string; figures: Record<string, string> }) {
  const chart = METRICS.map((metric) => ({
    metric: LABELS[metric],
    ...Object.fromEntries(models.map((m) => [m.model, m[metric] != null ? Number(((m[metric] as number) * 100).toFixed(2)) : 0])),
  }))
  const best = (metric: (typeof METRICS)[number]) =>
    models.reduce<BenchmarkModel | null>((top, m) => (m[metric] != null && (top === null || (m[metric] as number) > (top[metric] as number)) ? m : top), null)?.model

  return (
    <>
      <Card title={`${dataset} test split`} subtitle={`${models[0]?.num_images ? `${num(models[0].num_images)} held-out test cities` : `${num(models[0]?.num_patches)} held-out patches`} · ${evaluation ?? ''} · threshold ${models[0]?.threshold} · evaluated ${dateTime(generated)}`}>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chart} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1c2942" vertical={false} />
              <XAxis dataKey="metric" stroke="#7186a8" fontSize={12} tickLine={false} />
              <YAxis domain={[0, 100]} stroke="#7186a8" fontSize={12} tickLine={false} unit="%" />
              <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }} contentStyle={{ background: '#0e1626', border: '1px solid #2a3a58', borderRadius: 8, fontSize: 12 }}
                formatter={(value) => `${Number(value).toFixed(2)}%`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {models.map((m) => (
                <Bar key={m.model} dataKey={m.model} name={m.display_name} fill={MODEL_COLORS[m.model] ?? '#60a5fa'} radius={[4, 4, 0, 0]} maxBarSize={44} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card className="mt-4" title="Benchmark table">
        <div className="overflow-x-auto">
          <table className="tabular w-full min-w-[760px] text-left text-sm">
            <thead className="text-xs uppercase tracking-wider text-ink-400">
              <tr className="border-b border-ink-700/70">
                <th className="px-3 py-2 font-medium">Model</th>
                {METRICS.map((m) => <th key={m} className="px-3 py-2 text-right font-medium">{LABELS[m]}</th>)}
                <th className="px-3 py-2 text-right font-medium">Params</th>
                <th className="px-3 py-2 text-right font-medium">Latency</th>
                <th className="px-3 py-2 text-right font-medium">Peak VRAM</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.model} className="border-b border-ink-800 last:border-0">
                  <td className="px-3 py-3">
                    <div className="flex items-center gap-2 font-medium text-ink-100">
                      <span className="h-2.5 w-2.5 rounded-sm" style={{ background: MODEL_COLORS[m.model] ?? '#60a5fa' }} /> {m.display_name}
                    </div>
                    <div className="mt-0.5 font-mono text-[11px] text-ink-500">{m.checkpoint}</div>
                  </td>
                  {METRICS.map((metric) => (
                    <td key={metric} className={`px-3 py-3 text-right ${best(metric) === m.model ? 'font-semibold text-accent-300' : 'text-ink-200'}`}>{pct(m[metric])}</td>
                  ))}
                  <td className="px-3 py-3 text-right text-ink-200">{params(m.parameters)}</td>
                  <td className="px-3 py-3 text-right text-ink-200">{m.latency_ms_median.toFixed(1)} ms</td>
                  <td className="px-3 py-3 text-right text-ink-200">{m.peak_gpu_mb != null ? `${num(m.peak_gpu_mb)} MB` : 'n/a'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-ink-500">
          Metrics are for the change class over all test pixels. Latency is the median forward pass for one 256×256 bitemporal tile (batch 1) on{' '}
          {models[0]?.gpu_name ?? models[0]?.device}. Best value per column is highlighted.
        </p>
      </Card>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        {models.map((m) => {
          const [[tn, fp], [fn, tp]] = m.confusion_matrix.matrix
          return (
            <Card key={m.model} title={`Confusion matrix — ${m.display_name}`} subtitle="Pixels, rows = ground truth">
              <table className="tabular w-full border-separate border-spacing-1 text-center text-sm">
                <thead><tr><th /><th className="text-xs font-medium text-ink-400">Pred: no change</th><th className="text-xs font-medium text-ink-400">Pred: change</th></tr></thead>
                <tbody>
                  <tr><th className="pr-2 text-right text-xs font-medium text-ink-400">True: no change</th>
                    <td className="rounded bg-ink-700/70 px-2 py-3">{num(tn)}</td><td className="rounded bg-bad/15 px-2 py-3">{num(fp)}</td></tr>
                  <tr><th className="pr-2 text-right text-xs font-medium text-ink-400">True: change</th>
                    <td className="rounded bg-info/15 px-2 py-3">{num(fn)}</td><td className="rounded bg-ok/15 px-2 py-3">{num(tp)}</td></tr>
                </tbody>
              </table>
              {figures[`${m.model}_examples`] && (
                <details className="mt-4">
                  <summary className="cursor-pointer text-sm text-accent-300">Show test-split prediction examples</summary>
                  <img src={figures[`${m.model}_examples`]} alt={`${m.display_name} prediction examples`} loading="lazy" className="mt-3 w-full rounded-lg border border-ink-700" />
                </details>
              )}
            </Card>
          )
        })}
      </div>
    </>
  )
}

function SampleRunTable({ run }: { run: SampleRun }) {
  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-ink-400">
        <Tag>sample data</Tag>
        <span>{run.samples.length} labelled pairs · {run.threshold ? `threshold ${run.threshold}` : 'each model at its validation-selected threshold'} · {dateTime(run.created_at)}</span>
      </div>
      <div className="overflow-x-auto">
        <table className="tabular w-full min-w-[560px] text-left text-sm">
          <thead className="text-xs uppercase tracking-wider text-ink-400">
            <tr className="border-b border-ink-700/70">
              <th className="px-3 py-2 font-medium">Model</th>
              <th className="px-3 py-2 text-right font-medium">Precision</th>
              <th className="px-3 py-2 text-right font-medium">Recall</th>
              <th className="px-3 py-2 text-right font-medium">F1</th>
              <th className="px-3 py-2 text-right font-medium">IoU</th>
              <th className="px-3 py-2 text-right font-medium">Mean time / pair</th>
            </tr>
          </thead>
          <tbody>
            {run.models.map((m) => (
              <tr key={m.model} className="border-b border-ink-800 last:border-0">
                <td className="px-3 py-2.5 font-medium text-ink-100">
                  {m.display_name}
                  {m.dataset && <div className="text-xs font-normal text-ink-500">{m.dataset} samples · {m.samples?.length ?? 0} pairs{m.threshold != null ? ` · threshold ${m.threshold}` : ''}</div>}
                </td>
                <td className="px-3 py-2.5 text-right">{pct(m.precision)}</td>
                <td className="px-3 py-2.5 text-right">{pct(m.recall)}</td>
                <td className="px-3 py-2.5 text-right">{pct(m.f1)}</td>
                <td className="px-3 py-2.5 text-right">{pct(m.iou)}</td>
                <td className="px-3 py-2.5 text-right">{(m.mean_inference_seconds * 1000).toFixed(0)} ms ({m.device})</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-ink-500">{run.scope}. Counts are pooled over all pixels of the listed pairs.</p>
    </>
  )
}
