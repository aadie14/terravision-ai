import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError } from './api'

export interface AsyncState<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

function toApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError('UNKNOWN', err instanceof Error ? err.message : String(err), 0)
}

/** Run an async loader on mount / when deps change; optionally re-poll every `pollMs`. */
export function useAsync<T>(loader: () => Promise<T>, deps: unknown[], pollMs?: number): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)
  const loaderRef = useRef(loader)
  loaderRef.current = loader

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const run = (initial: boolean) => {
      if (initial) setLoading(true)
      loaderRef
        .current()
        .then((value) => {
          if (cancelled) return
          setData(value)
          setError(null)
        })
        .catch((err: unknown) => {
          if (!cancelled) setError(toApiError(err))
        })
        .finally(() => {
          if (cancelled) return
          setLoading(false)
          if (pollMs) timer = setTimeout(() => run(false), pollMs)
        })
    }
    run(true)
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick, pollMs])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, error, loading, reload }
}

const PREFS_KEY = 'terravision.prefs.v1'

export interface Prefs {
  defaultModel: string | null
  overlayOpacity: number
  autoRefreshSeconds: number
}

const DEFAULT_PREFS: Prefs = { defaultModel: null, overlayOpacity: 0.6, autoRefreshSeconds: 5 }

export function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(PREFS_KEY)
    return raw ? { ...DEFAULT_PREFS, ...(JSON.parse(raw) as Partial<Prefs>) } : DEFAULT_PREFS
  } catch {
    return DEFAULT_PREFS
  }
}

export function usePrefs(): [Prefs, (patch: Partial<Prefs>) => void] {
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs)
  const update = useCallback((patch: Partial<Prefs>) => {
    setPrefs((current) => {
      const next = { ...current, ...patch }
      try {
        localStorage.setItem(PREFS_KEY, JSON.stringify(next))
      } catch {
        // storage unavailable: preferences simply do not persist
      }
      return next
    })
  }, [])
  return [prefs, update]
}

export { toApiError }
