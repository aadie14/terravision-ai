import { History as HistoryIcon } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'

import { AnalysisTable } from '../components/AnalysisTable'
import { useToast } from '../components/Toasts'
import { Button, Card, cx, EmptyState, ErrorBanner, PageHeader, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { toApiError, useAsync } from '../lib/hooks'
import type { AnalysisStatus, AnalysisSummary } from '../lib/types'

const PAGE = 20
const FILTERS: (AnalysisStatus | 'all')[] = ['all', 'completed', 'running', 'queued', 'failed']

export function HistoryPage() {
  const { notify } = useToast()
  const [filter, setFilter] = useState<AnalysisStatus | 'all'>('all')
  const [page, setPage] = useState(0)
  const [pending, setPending] = useState<AnalysisSummary | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const { data, error, loading, reload } = useAsync(
    () => api.listAnalyses({ limit: PAGE, offset: page * PAGE, status: filter === 'all' ? undefined : filter }),
    [filter, page],
    6000,
  )

  const confirmDelete = async () => {
    if (!pending) return
    setDeleting(pending.id)
    try {
      await api.deleteAnalysis(pending.id)
      notify('success', 'Analysis and its files were deleted.')
      reload()
    } catch (err) {
      notify('error', toApiError(err).message)
    } finally {
      setDeleting(null)
      setPending(null)
    }
  }

  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE)) : 1

  return (
    <>
      <PageHeader title="Analysis history" description="Every analysis stored in the local database, newest first." />
      <div className="mb-4 flex flex-wrap gap-1.5" role="tablist" aria-label="Filter by status">
        {FILTERS.map((f) => (
          <button key={f} type="button" role="tab" aria-selected={filter === f} onClick={() => { setFilter(f); setPage(0) }}
            className={cx('rounded-md px-3 py-1.5 text-xs font-medium capitalize transition-colors',
              filter === f ? 'bg-accent-500/15 text-accent-300' : 'text-ink-300 hover:bg-ink-700/70')}>
            {f}
          </button>
        ))}
      </div>

      {loading && !data ? <Spinner /> : error && !data ? <ErrorBanner error={error} onRetry={reload} /> : data && data.items.length === 0 ? (
        <EmptyState icon={<HistoryIcon className="h-8 w-8" />} title={filter === 'all' ? 'No analyses yet' : `No ${filter} analyses`}
          action={filter === 'all' ? <Link to="/new"><Button variant="primary">New analysis</Button></Link> : undefined}>
          {filter === 'all' ? 'Analyses you run are listed here with their status, model and result.' : 'Try another status filter.'}
        </EmptyState>
      ) : data ? (
        <Card>
          <AnalysisTable items={data.items} onDelete={setPending} deletingId={deleting} />
          <div className="mt-4 flex items-center justify-between text-xs text-ink-400">
            <span>{data.total} analyses · page {page + 1} of {pages}</span>
            <div className="flex gap-2">
              <Button disabled={page === 0} onClick={() => setPage((p) => p - 1)}>Previous</Button>
              <Button disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)}>Next</Button>
            </div>
          </div>
        </Card>
      ) : null}

      {pending && (
        <div className="fixed inset-0 z-[1500] flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal="true" aria-labelledby="del-title">
          <div className="w-full max-w-md rounded-xl border border-ink-600 bg-ink-850 p-5 shadow-2xl">
            <h2 id="del-title" className="text-base font-semibold text-ink-100">Delete this analysis?</h2>
            <p className="mt-2 text-sm text-ink-300">
              “{pending.name || pending.id.slice(0, 8)}” and all of its stored images and results will be permanently removed.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <Button onClick={() => setPending(null)}>Cancel</Button>
              <Button variant="danger" loading={deleting !== null} onClick={confirmDelete}>Delete</Button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
