import { Activity, Boxes, CheckCircle2, Cpu, Layers, Percent, PlusCircle, XCircle } from 'lucide-react'
import { Link } from 'react-router-dom'

import { AnalysisTable } from '../components/AnalysisTable'
import { Button, Card, EmptyState, ErrorBanner, PageHeader, Spinner, StatCard } from '../components/ui'
import { api } from '../lib/api'
import { num, pct, seconds } from '../lib/format'
import { useAsync } from '../lib/hooks'

export function OverviewPage() {
  const { data, error, loading, reload } = useAsync(
    () => Promise.all([api.stats(), api.models(), api.system(), api.listAnalyses({ limit: 8 })]),
    [],
    8000,
  )

  if (loading && !data) return <Spinner label="Loading dashboard…" />
  if (error && !data) return <ErrorBanner error={error} onRetry={reload} title="Dashboard unavailable" />
  if (!data) return null
  const [stats, models, system, recent] = data
  const available = models.models.filter((m) => m.available)
  const gpu = system.compute.device === 'cuda'

  return (
    <>
      <PageHeader
        title="Overview"
        description="Live state of the change-detection service: analyses run on this machine, model availability and compute status."
        actions={
          <Link to="/new">
            <Button variant="primary">
              <PlusCircle className="h-4 w-4" aria-hidden /> New analysis
            </Button>
          </Link>
        }
      />
      {error && <div className="mb-4"><ErrorBanner error={error} onRetry={reload} title="Live refresh failed — showing last known data" /></div>}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Total analyses" value={num(stats.total)} icon={<Layers className="h-4 w-4" />}
          hint={stats.in_progress ? `${stats.in_progress} in progress` : 'none in progress'} />
        <StatCard label="Completed" value={num(stats.completed)} tone="ok" icon={<CheckCircle2 className="h-4 w-4" />}
          hint={stats.mean_inference_seconds != null ? `mean inference ${seconds(stats.mean_inference_seconds)}` : 'no completed runs yet'} />
        <StatCard label="Failed" value={num(stats.failed)} tone={stats.failed ? 'bad' : 'default'} icon={<XCircle className="h-4 w-4" />}
          hint="validation or inference errors" />
        <StatCard label="Changed area" tone="accent" icon={<Percent className="h-4 w-4" />}
          value={stats.mean_changed_percent != null ? `${stats.mean_changed_percent.toFixed(2)}%` : 'n/a'}
          hint={stats.last_changed_percent != null ? `mean over completed · latest ${stats.last_changed_percent.toFixed(2)}%` : 'no completed analyses yet'} />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Models" subtitle={`${available.length} of ${models.models.length} available`}
          action={<Link to="/compare" className="text-xs text-accent-300 hover:underline">Compare</Link>}>
          <ul className="space-y-3">
            {models.models.map((m) => (
              <li key={m.key} className="flex items-start gap-3">
                <Boxes className={`mt-0.5 h-4 w-4 shrink-0 ${m.available ? 'text-accent-400' : 'text-ink-500'}`} aria-hidden />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2 text-sm">
                    <span className="truncate font-medium text-ink-100">{m.display_name}</span>
                    <span className={m.available ? 'text-xs text-ok' : 'text-xs text-bad'}>{m.available ? 'ready' : 'unavailable'}</span>
                  </div>
                  <p className="mt-0.5 text-xs text-ink-400">
                    {m.benchmark?.f1 != null
                      ? `${m.benchmark.dataset ?? 'Test'} test F1 ${pct(m.benchmark.f1, 2)} · IoU ${pct(m.benchmark.iou, 2)}`
                      : m.available ? 'Not benchmarked yet' : (m.unavailable_reason ?? 'No checkpoint')}
                  </p>
                </div>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Compute" subtitle="Inference device used by the API">
          <div className="flex items-center gap-3">
            <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${gpu ? 'bg-accent-500/15 text-accent-300' : 'bg-warn/15 text-warn'}`}>
              <Cpu className="h-5 w-5" aria-hidden />
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-ink-100">{gpu ? system.compute.gpu_name : 'CPU (no CUDA device)'}</div>
              <div className="text-xs text-ink-400">
                {gpu ? `CUDA ${system.compute.cuda_version} · mixed precision ${system.compute.amp ? 'on' : 'off'}` : 'GPU acceleration unavailable — inference is slower'}
              </div>
            </div>
          </div>
          {gpu && system.compute.gpu_memory_total_mb != null && system.compute.gpu_memory_free_mb != null && (
            <div className="mt-4">
              <div className="mb-1 flex justify-between text-xs text-ink-400">
                <span>GPU memory in use</span>
                <span className="tabular">
                  {num(system.compute.gpu_memory_total_mb - system.compute.gpu_memory_free_mb)} / {num(system.compute.gpu_memory_total_mb)} MB
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-ink-700">
                <div className="h-full rounded-full bg-accent-400"
                  style={{ width: `${Math.min(100, ((system.compute.gpu_memory_total_mb - system.compute.gpu_memory_free_mb) / system.compute.gpu_memory_total_mb) * 100)}%` }} />
              </div>
            </div>
          )}
          <p className="mt-4 text-xs text-ink-500">PyTorch {system.compute.torch} · Python {system.platform.python} · {system.platform.os}</p>
        </Card>

        <Card title="Runs per model" subtitle="All analyses stored in the database">
          {Object.keys(stats.by_model).length === 0 ? (
            <p className="text-sm text-ink-400">No analyses have been run yet.</p>
          ) : (
            <ul className="space-y-3">
              {models.models.map((m) => {
                const count = stats.by_model[m.key] ?? 0
                return (
                  <li key={m.key}>
                    <div className="mb-1 flex justify-between text-xs text-ink-300">
                      <span>{m.display_name}</span>
                      <span className="tabular">{count}</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-ink-700">
                      <div className={`h-full rounded-full ${m.family === 'transformer' ? 'bg-accent-400' : 'bg-amber-500'}`}
                        style={{ width: `${stats.total ? (count / stats.total) * 100 : 0}%` }} />
                    </div>
                  </li>
                )
              })}
            </ul>
          )}
        </Card>
      </div>

      <Card className="mt-4" title="Recent analyses"
        action={<Link to="/history" className="text-xs text-accent-300 hover:underline">View all</Link>}>
        {recent.items.length === 0 ? (
          <EmptyState icon={<Activity className="h-8 w-8" />} title="No analyses yet"
            action={<Link to="/new"><Button variant="primary">Run your first analysis</Button></Link>}>
            Upload two co-registered satellite images, pick a bundled LEVIR-CD sample, or fetch Sentinel-2 scenes from the map.
          </EmptyState>
        ) : (
          <AnalysisTable items={recent.items} />
        )}
      </Card>
    </>
  )
}
