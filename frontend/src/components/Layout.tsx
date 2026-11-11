import {
  Activity,
  BarChart3,
  BookOpen,
  Cpu,
  History,
  Layers,
  LayoutDashboard,
  Map as MapIcon,
  Menu,
  PlusCircle,
  Settings,
  X,
} from 'lucide-react'
import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

import { api } from '../lib/api'
import { useAsync } from '../lib/hooks'
import { cx } from './ui'

// eslint-disable-next-line react-refresh/only-export-components
export const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/new', label: 'New Analysis', icon: PlusCircle },
  { to: '/map', label: 'Satellite Map', icon: MapIcon },
  { to: '/results', label: 'Change Results', icon: Layers },
  { to: '/compare', label: 'Model Comparison', icon: BarChart3 },
  { to: '/history', label: 'Analysis History', icon: History },
  { to: '/research', label: 'Research & Models', icon: BookOpen },
  { to: '/settings', label: 'System Status', icon: Settings },
]

function BackendStatus() {
  const { data, error } = useAsync(() => Promise.all([api.health(), api.system()]), [], 15000)
  if (error) {
    return (
      <div className="rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad" role="status">
        <div className="flex items-center gap-2 font-medium">
          <span className="h-2 w-2 rounded-full bg-bad" /> Backend offline
        </div>
        <p className="mt-1 text-ink-300">Start the API on port 8000.</p>
      </div>
    )
  }
  if (!data) return <div className="h-14 animate-pulse rounded-lg bg-ink-800" aria-hidden />
  const [health, system] = data
  const gpu = system.compute.device === 'cuda'
  return (
    <div className="rounded-lg border border-ink-700 bg-ink-900/70 px-3 py-2 text-xs" role="status">
      <div className="flex items-center gap-2 font-medium text-ink-100">
        <span className={cx('h-2 w-2 rounded-full', health.status === 'ok' ? 'bg-ok' : 'bg-warn')} />
        API {health.status === 'ok' ? 'online' : 'degraded'}
        <span className="ml-auto font-mono text-[10px] text-ink-500">v{health.version}</span>
      </div>
      <div className="mt-1.5 flex items-center gap-1.5 text-ink-300">
        <Cpu className="h-3.5 w-3.5 text-ink-500" aria-hidden />
        <span className="truncate">{gpu ? (system.compute.gpu_name ?? 'CUDA GPU') : 'CPU inference'}</span>
      </div>
    </div>
  )
}

export function Layout() {
  const [open, setOpen] = useState(false)
  return (
    <div className="flex h-full">
      {open && <div className="fixed inset-0 z-40 bg-black/60 lg:hidden" onClick={() => setOpen(false)} aria-hidden />}
      <aside
        className={cx(
          'fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-ink-700/70 bg-ink-900 transition-transform lg:static lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
        aria-label="Primary"
      >
        <div className="flex items-center gap-3 px-5 py-5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-500/15 text-accent-300">
            <Activity className="h-5 w-5" aria-hidden />
          </div>
          <div className="min-w-0">
            <div className="text-sm font-semibold text-ink-100">TerraVision AI</div>
            <div className="truncate text-[11px] text-ink-400">Satellite Change Intelligence</div>
          </div>
          <button type="button" className="ml-auto text-ink-400 lg:hidden" aria-label="Close navigation" onClick={() => setOpen(false)}>
            <X className="h-5 w-5" aria-hidden />
          </button>
        </div>
        <nav className="flex-1 space-y-0.5 overflow-y-auto px-3">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                cx(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
                  isActive ? 'bg-accent-500/12 font-medium text-accent-300' : 'text-ink-300 hover:bg-ink-800 hover:text-ink-100',
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" aria-hidden />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="space-y-2 p-3">
          <BackendStatus />
          <p className="px-1 text-[10px] leading-relaxed text-ink-500">
            Research prototype · Phase 1
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-ink-700/70 bg-ink-900 px-4 py-3 lg:hidden">
          <button type="button" aria-label="Open navigation" onClick={() => setOpen(true)} className="text-ink-200">
            <Menu className="h-5 w-5" aria-hidden />
          </button>
          <span className="text-sm font-semibold">TerraVision AI</span>
        </header>
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[1400px] px-4 py-6 sm:px-6 lg:px-8">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
