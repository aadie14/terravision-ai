import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

import type { ApiError } from '../lib/api'
import type { AnalysisStatus } from '../lib/types'

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}

export function Card({ children, className, title, action, subtitle }: {
  children: ReactNode
  className?: string
  title?: ReactNode
  subtitle?: ReactNode
  action?: ReactNode
}) {
  return (
    <section className={cx('rounded-xl border border-ink-700/70 bg-ink-850/80 shadow-lg shadow-black/20', className)}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-3 border-b border-ink-700/60 px-5 py-3.5">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold tracking-wide text-ink-100">{title}</h2>
            {subtitle && <p className="mt-0.5 text-xs text-ink-400">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}

export function StatCard({ label, value, hint, icon, tone = 'default' }: {
  label: string
  value: ReactNode
  hint?: ReactNode
  icon?: ReactNode
  tone?: 'default' | 'ok' | 'bad' | 'accent' | 'warn'
}) {
  const tones = {
    default: 'text-ink-100',
    ok: 'text-ok',
    bad: 'text-bad',
    accent: 'text-accent-300',
    warn: 'text-warn',
  }
  return (
    <div className="rounded-xl border border-ink-700/70 bg-ink-850/80 p-4">
      <div className="flex items-center justify-between text-xs font-medium uppercase tracking-wider text-ink-400">
        <span>{label}</span>
        {icon && <span className="text-ink-500">{icon}</span>}
      </div>
      <div className={cx('tabular mt-2 text-2xl font-semibold', tones[tone])} data-testid={`stat-${label}`}>
        {value}
      </div>
      {hint && <div className="mt-1 text-xs text-ink-400">{hint}</div>}
    </div>
  )
}

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

export function Button({ variant = 'secondary', loading, className, children, disabled, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; loading?: boolean }) {
  const variants: Record<ButtonVariant, string> = {
    primary: 'bg-accent-500 text-ink-950 hover:bg-accent-400 font-semibold',
    secondary: 'bg-ink-700 text-ink-100 hover:bg-ink-600 border border-ink-600',
    ghost: 'text-ink-200 hover:bg-ink-700/70',
    danger: 'bg-bad/15 text-bad hover:bg-bad/25 border border-bad/30',
  }
  return (
    <button
      type="button"
      {...rest}
      disabled={disabled || loading}
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm transition-colors',
        'disabled:cursor-not-allowed disabled:opacity-50',
        variants[variant],
        className,
      )}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  )
}

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div role="status" className="flex items-center justify-center gap-3 py-10 text-sm text-ink-300">
      <Loader2 className="h-5 w-5 animate-spin text-accent-400" aria-hidden />
      <span>{label}</span>
    </div>
  )
}

export function EmptyState({ icon, title, children, action }: {
  icon?: ReactNode
  title: string
  children?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-ink-600 px-6 py-12 text-center">
      {icon && <div className="mb-3 text-ink-500">{icon}</div>}
      <h3 className="text-base font-semibold text-ink-100">{title}</h3>
      {children && <p className="mt-1.5 max-w-md text-sm text-ink-400">{children}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}

export function ErrorBanner({ error, onRetry, title }: { error: ApiError | string; onRetry?: () => void; title?: string }) {
  const message = typeof error === 'string' ? error : error.message
  const code = typeof error === 'string' ? null : error.code
  return (
    <div role="alert" className="flex items-start gap-3 rounded-xl border border-bad/40 bg-bad/10 px-4 py-3 text-sm">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-bad" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="font-medium text-ink-100">{title ?? 'Something went wrong'}</p>
        <p className="mt-0.5 break-words text-ink-300">{message}</p>
        {code && <p className="mt-1 font-mono text-[11px] text-ink-500">{code}</p>}
      </div>
      {onRetry && (
        <Button variant="ghost" onClick={onRetry} className="shrink-0">
          <RefreshCw className="h-4 w-4" aria-hidden /> Retry
        </Button>
      )}
    </div>
  )
}

export function WarningList({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return null
  return (
    <div role="note" className="rounded-xl border border-warn/40 bg-warn/10 px-4 py-3 text-sm text-ink-200">
      <div className="mb-1 flex items-center gap-2 font-medium text-warn">
        <AlertTriangle className="h-4 w-4" aria-hidden /> Notes about this analysis
      </div>
      <ul className="list-disc space-y-1 pl-5">
        {warnings.map((w) => (
          <li key={w}>{w}</li>
        ))}
      </ul>
    </div>
  )
}

const STATUS_STYLES: Record<AnalysisStatus, string> = {
  queued: 'bg-ink-600/60 text-ink-200',
  running: 'bg-info/15 text-info',
  completed: 'bg-ok/15 text-ok',
  failed: 'bg-bad/15 text-bad',
}

export function StatusBadge({ status }: { status: AnalysisStatus }) {
  return (
    <span className={cx('inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium', STATUS_STYLES[status])}>
      {status === 'running' && <Loader2 className="h-3 w-3 animate-spin" aria-hidden />}
      {status}
    </span>
  )
}

export function Tag({ children, tone = 'default' }: { children: ReactNode; tone?: 'default' | 'accent' | 'warn' }) {
  const tones = {
    default: 'border-ink-600 text-ink-300',
    accent: 'border-accent-500/50 text-accent-300',
    warn: 'border-warn/50 text-warn',
  }
  return <span className={cx('rounded-md border px-1.5 py-0.5 text-[11px] font-medium', tones[tone])}>{children}</span>
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink-100">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-ink-400">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

export function Field({ label, hint, children, htmlFor }: { label: string; hint?: ReactNode; children: ReactNode; htmlFor?: string }) {
  return (
    <div>
      <label htmlFor={htmlFor} className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-ink-400">
        {label}
      </label>
      {children}
      {hint && <p className="mt-1 text-xs text-ink-500">{hint}</p>}
    </div>
  )
}

export const inputClass =
  'w-full rounded-lg border border-ink-600 bg-ink-900 px-3 py-2 text-sm text-ink-100 placeholder:text-ink-500 focus:border-accent-500 focus:outline-none'

export function KeyValue({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-ink-400">{k}</dt>
          <dd className="tabular min-w-0 break-words text-right text-ink-100">{v}</dd>
        </div>
      ))}
    </dl>
  )
}
