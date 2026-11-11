import { useState } from 'react'

/** Before/after reveal slider. Keyboard accessible through the native range input. */
export function CompareSlider({ before, after, beforeLabel = 'T1 · before', afterLabel = 'T2 · after', overlay }: {
  before: string
  after: string
  beforeLabel?: string
  afterLabel?: string
  overlay?: { src: string; opacity: number } | null
}) {
  const [position, setPosition] = useState(50)
  return (
    <div className="relative select-none overflow-hidden rounded-xl border border-ink-700 bg-ink-950">
      <img src={after} alt={afterLabel} className="block w-full" draggable={false} />
      {overlay && (
        <img src={overlay.src} alt="" aria-hidden className="pixelated pointer-events-none absolute inset-0 h-full w-full" style={{ opacity: overlay.opacity }} />
      )}
      <img
        src={before}
        alt={beforeLabel}
        draggable={false}
        className="absolute inset-0 h-full w-full"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      />
      <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/90 shadow-[0_0_0_1px_rgba(0,0,0,0.4)]" style={{ left: `${position}%` }}>
        <div className="absolute left-1/2 top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-xs font-bold text-ink-900 shadow-lg">
          ⇆
        </div>
      </div>
      <span className="pointer-events-none absolute left-3 top-3 rounded bg-ink-950/80 px-2 py-0.5 text-xs text-ink-100">{beforeLabel}</span>
      <span className="pointer-events-none absolute right-3 top-3 rounded bg-ink-950/80 px-2 py-0.5 text-xs text-ink-100">{afterLabel}</span>
      <input
        type="range"
        min={0}
        max={100}
        step={0.5}
        value={position}
        onChange={(e) => setPosition(Number(e.target.value))}
        aria-label="Before / after comparison position"
        className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0"
      />
    </div>
  )
}
