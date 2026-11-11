import { CheckCircle2, RefreshCw, XCircle } from 'lucide-react'

import { Button, Card, ErrorBanner, Field, inputClass, KeyValue, PageHeader, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { bytes, dateTime, num } from '../lib/format'
import { useAsync, usePrefs } from '../lib/hooks'

function Check({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 ${ok ? 'text-ok' : 'text-bad'}`}>
      {ok ? <CheckCircle2 className="h-4 w-4" aria-hidden /> : <XCircle className="h-4 w-4" aria-hidden />} {label}
    </span>
  )
}

export function SettingsPage() {
  const { data, error, loading, reload } = useAsync(
    () => Promise.all([api.health(), api.system(), api.config(), api.models(), api.providers()]),
    [],
  )
  const [prefs, setPrefs] = usePrefs()

  return (
    <>
      <PageHeader title="Settings & system status" description="Backend health, compute device, limits and local display preferences."
        actions={<Button onClick={reload}><RefreshCw className="h-4 w-4" aria-hidden /> Refresh</Button>} />

      {loading && !data ? <Spinner label="Checking backend…" /> : error || !data ? (
        <ErrorBanner error={error ?? 'No data'} onRetry={reload} title="Backend unreachable" />
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Backend health" subtitle={`Checked ${dateTime(data[0].time)}`}>
            <KeyValue rows={[
              ['API', <Check key="a" ok={data[0].status === 'ok'} label={data[0].status} />],
              ['Database', <Check key="d" ok={data[0].database} label={`${data[1].database.engine} ${data[0].database ? 'connected' : 'unreachable'}`} />],
              ['Models ready', <Check key="m" ok={data[0].models_available > 0} label={`${data[0].models_available} of ${data[3].models.length}`} />],
              ['Version', data[0].version],
              ['Stored results', bytes(data[1].storage.analyses_bytes)],
              ['Free disk space', bytes(data[1].storage.disk_free_bytes)],
              ['Bundled samples', num(data[1].samples)],
            ]} />
          </Card>

          <Card title="Compute">
            <KeyValue rows={[
              ['Device in use', data[1].compute.device === 'cuda' ? `GPU — ${data[1].compute.gpu_name}` : 'CPU'],
              ['CUDA available', <Check key="c" ok={data[1].compute.cuda_available} label={data[1].compute.cuda_available ? `yes (CUDA ${data[1].compute.cuda_version})` : 'no — CPU fallback active'} />],
              ['Mixed precision', data[1].compute.amp ? 'enabled' : 'disabled'],
              ['GPU memory', data[1].compute.gpu_memory_total_mb != null ? `${num(data[1].compute.gpu_memory_free_mb)} MB free of ${num(data[1].compute.gpu_memory_total_mb)} MB` : 'n/a'],
              ['Allocated by API', data[1].compute.gpu_memory_allocated_mb != null ? `${num(data[1].compute.gpu_memory_allocated_mb)} MB` : 'n/a'],
              ['PyTorch', data[1].compute.torch],
              ['Python / OS', `${data[1].platform.python} · ${data[1].platform.os}`],
            ]} />
          </Card>

          <Card title="Models">
            <ul className="space-y-3 text-sm">
              {data[3].models.map((m) => (
                <li key={m.key} className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="font-medium text-ink-100">{m.display_name}</div>
                    <div className="truncate font-mono text-[11px] text-ink-500">{m.checkpoint?.version ?? m.checkpoint_file}</div>
                    {!m.available && <div className="text-xs text-bad">{m.unavailable_reason}</div>}
                  </div>
                  <div className="shrink-0 text-right text-xs">
                    <Check ok={m.available} label={m.available ? 'available' : 'unavailable'} />
                    <div className="mt-0.5 text-ink-500">{m.loaded ? 'in memory' : 'loads on first use'}</div>
                  </div>
                </li>
              ))}
            </ul>
          </Card>

          <Card title="Satellite providers">
            <ul className="space-y-3 text-sm">
              {data[4].providers.map((p) => (
                <li key={p.key}>
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-medium text-ink-100">{p.name}</span>
                    <Check ok={p.configured} label={p.configured ? 'ready' : 'credentials required'} />
                  </div>
                  <p className="mt-0.5 text-xs text-ink-400">{p.data_license}</p>
                  {!p.verified && p.requires_credentials && (
                    <p className="mt-0.5 text-xs text-warn">Retrieval implemented against the documented API but not verified end-to-end (no credentials available).</p>
                  )}
                </li>
              ))}
            </ul>
          </Card>

          <Card title="Server limits" subtitle="Configured through TV_* environment variables">
            <KeyValue rows={[
              ['Max upload size', `${data[2].upload.max_file_mb} MB per file`],
              ['Image side', `${data[2].upload.min_image_side}–${data[2].upload.max_image_side} px`],
              ['File types', data[2].upload.allowed_extensions.join(' ')],
              ['Default threshold', data[2].inference.default_threshold.toFixed(2)],
              ['Tile overlap', `${data[2].inference.tile_overlap} px`],
              ['AOI side', `${data[2].satellite.min_aoi_km}–${data[2].satellite.max_aoi_km} km`],
            ]} />
          </Card>

          <Card title="Display preferences" subtitle="Stored in this browser only">
            <div className="space-y-4">
              <Field label="Preferred model for new analyses" htmlFor="pref-model">
                <select id="pref-model" className={inputClass} value={prefs.defaultModel ?? ''}
                  onChange={(e) => setPrefs({ defaultModel: e.target.value || null })}>
                  <option value="">Server default ({data[3].default})</option>
                  {data[3].models.map((m) => <option key={m.key} value={m.key} disabled={!m.available}>{m.display_name}</option>)}
                </select>
              </Field>
              <Field label={`Default overlay opacity · ${Math.round(prefs.overlayOpacity * 100)}%`} htmlFor="pref-opacity">
                <input id="pref-opacity" type="range" min={0.1} max={1} step={0.05} value={prefs.overlayOpacity}
                  onChange={(e) => setPrefs({ overlayOpacity: Number(e.target.value) })} className="w-full accent-teal-400" />
              </Field>
            </div>
          </Card>
        </div>
      )}
    </>
  )
}
