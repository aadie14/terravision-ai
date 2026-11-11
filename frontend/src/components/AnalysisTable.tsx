import { MapPin, Target, Trash2 } from 'lucide-react'
import { Link } from 'react-router-dom'

import { dateTime, modelLabel, pct, seconds } from '../lib/format'
import type { AnalysisSummary } from '../lib/types'
import { StatusBadge } from './ui'

export function AnalysisTable({ items, onDelete, deletingId }: {
  items: AnalysisSummary[]
  onDelete?: (item: AnalysisSummary) => void
  deletingId?: string | null
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-left text-sm">
        <thead className="text-xs uppercase tracking-wider text-ink-400">
          <tr className="border-b border-ink-700/70">
            <th className="px-3 py-2 font-medium">Analysis</th>
            <th className="px-3 py-2 font-medium">Model</th>
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="px-3 py-2 text-right font-medium">Changed</th>
            <th className="px-3 py-2 text-right font-medium">Inference</th>
            <th className="px-3 py-2 font-medium">Created</th>
            {onDelete && <th className="px-3 py-2" aria-label="Actions" />}
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.id} className="border-b border-ink-800 last:border-0 hover:bg-ink-800/40">
              <td className="max-w-[320px] px-3 py-2.5">
                <Link to={`/results/${item.id}`} className="block truncate font-medium text-ink-100 hover:text-accent-300">
                  {item.name || item.id.slice(0, 8)}
                </Link>
                <div className="mt-0.5 flex items-center gap-2 text-xs text-ink-500">
                  <span className="font-mono">{item.id.slice(0, 8)}</span>
                  <span>{item.source}</span>
                  {item.width && item.height && <span>{item.width}×{item.height}</span>}
                  {item.georeferenced && <MapPin className="h-3 w-3 text-accent-400" aria-label="Georeferenced" />}
                  {item.has_ground_truth && <Target className="h-3 w-3 text-info" aria-label="Has ground truth" />}
                </div>
              </td>
              <td className="px-3 py-2.5 text-ink-300">{modelLabel(item.model_key)}</td>
              <td className="px-3 py-2.5">
                <StatusBadge status={item.status} />
              </td>
              <td className="tabular px-3 py-2.5 text-right text-ink-200">{pct(item.changed_fraction)}</td>
              <td className="tabular px-3 py-2.5 text-right text-ink-200">{seconds(item.inference_seconds)}</td>
              <td className="whitespace-nowrap px-3 py-2.5 text-ink-400">{dateTime(item.created_at)}</td>
              {onDelete && (
                <td className="px-3 py-2.5 text-right">
                  <button
                    type="button"
                    onClick={() => onDelete(item)}
                    disabled={deletingId === item.id}
                    aria-label={`Delete analysis ${item.name || item.id.slice(0, 8)}`}
                    className="rounded p-1.5 text-ink-400 hover:bg-bad/15 hover:text-bad disabled:opacity-40"
                  >
                    <Trash2 className="h-4 w-4" aria-hidden />
                  </button>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
