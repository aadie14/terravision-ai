import { CheckCircle2, XCircle } from 'lucide-react'

import { DOMAIN_SHORT, params, pct } from '../lib/format'
import type { ModelInfo } from '../lib/types'
import { cx, Tag } from './ui'

/** Radio-card model picker. Models without a trained checkpoint are shown but cannot be selected. */
export function ModelSelect({ models, value, onChange }: {
  models: ModelInfo[]
  value: string
  onChange: (key: string) => void
}) {
  return (
    <div role="radiogroup" aria-label="Change-detection model" className="grid gap-3 sm:grid-cols-2">
      {models.map((model) => {
        const selected = value === model.key
        return (
          <button
            key={model.key}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={!model.available}
            onClick={() => onChange(model.key)}
            className={cx(
              'rounded-xl border p-4 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-60',
              selected ? 'border-accent-400 bg-accent-500/10' : 'border-ink-600 bg-ink-900/60 hover:border-ink-500',
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-semibold text-ink-100">{model.display_name}</span>
              <span className="flex shrink-0 gap-1">
                <Tag>{DOMAIN_SHORT[model.domain] ?? model.domain}</Tag>
                <Tag tone={model.family === 'transformer' ? 'accent' : 'default'}>{model.family}</Tag>
              </span>
            </div>
            <p className="mt-1.5 line-clamp-2 text-xs text-ink-400">{model.architecture}</p>
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-300">
              {model.available ? (
                <span className="flex items-center gap-1 text-ok">
                  <CheckCircle2 className="h-3.5 w-3.5" aria-hidden /> trained checkpoint
                </span>
              ) : (
                <span className="flex items-center gap-1 text-bad">
                  <XCircle className="h-3.5 w-3.5" aria-hidden /> no trained checkpoint
                </span>
              )}
              <span>{params(model.parameters)} params</span>
              {model.benchmark?.f1 != null && <span>test F1 {pct(model.benchmark.f1, 1)}</span>}
            </div>
            {!model.available && model.unavailable_reason && (
              <p className="mt-2 text-[11px] text-ink-500">{model.unavailable_reason}</p>
            )}
          </button>
        )
      })}
    </div>
  )
}
