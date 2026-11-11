export const NA = 'n/a'

export function pct(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || Number.isNaN(value) ? NA : `${(value * 100).toFixed(digits)}%`
}

export function num(value: number | null | undefined, digits = 0): string {
  return value === null || value === undefined || Number.isNaN(value)
    ? NA
    : value.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

export function seconds(value: number | null | undefined): string {
  if (value === null || value === undefined) return NA
  return value < 1 ? `${Math.round(value * 1000)} ms` : `${value.toFixed(2)} s`
}

export function bytes(value: number): string {
  if (value < 1024) return `${value} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let v = value / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`
}

export function params(value: number | null | undefined): string {
  if (value === null || value === undefined) return NA
  return value >= 1e6 ? `${(value / 1e6).toFixed(2)} M` : `${(value / 1e3).toFixed(0)} K`
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return NA
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? NA
    : d.toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })
}

export function area(m2: number): string {
  if (m2 >= 1e6) return `${(m2 / 1e6).toFixed(2)} km²`
  if (m2 >= 1e4) return `${(m2 / 1e4).toFixed(2)} ha`
  return `${Math.round(m2).toLocaleString('en-US')} m²`
}

export const MODEL_LABELS: Record<string, string> = {
  swin_siamese: 'Siamese Swin Transformer',
  fc_siam_diff: 'FC-Siam-diff (CNN)',
  swin_siamese_s2: 'Siamese Swin (Sentinel-2)',
  fc_siam_diff_s2: 'FC-Siam-diff (Sentinel-2)',
}

export function modelLabel(key: string): string {
  return MODEL_LABELS[key] ?? key
}

export const MODEL_COLORS: Record<string, string> = {
  swin_siamese: '#2dd4bf',
  fc_siam_diff: '#f59e0b',
  swin_siamese_s2: '#38bdf8',
  fc_siam_diff_s2: '#f472b6',
}

export const DOMAIN_SHORT: Record<string, string> = { vhr: 'LEVIR-CD · 0.5 m', sentinel2: 'Sentinel-2 · 10 m' }
