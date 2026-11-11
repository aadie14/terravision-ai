import { ImagePlus, X } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'

import { bytes } from '../lib/format'
import { cx } from './ui'

const PREVIEWABLE = /\.(png|jpe?g)$/i

export interface FileDropProps {
  label: string
  hint?: string
  file: File | null
  onChange: (file: File | null) => void
  accept: string[]
  maxMb: number
  optional?: boolean
}

/** Client-side pre-check; the server re-validates everything (type, integrity, size). */
// eslint-disable-next-line react-refresh/only-export-components
export function validateFile(file: File, accept: string[], maxMb: number): string | null {
  const ext = file.name.includes('.') ? file.name.slice(file.name.lastIndexOf('.')).toLowerCase() : ''
  if (!accept.includes(ext)) return `Unsupported file type "${ext || 'unknown'}". Use ${accept.join(', ')}.`
  if (file.size === 0) return 'The file is empty.'
  if (file.size > maxMb * 1024 * 1024) return `The file is ${bytes(file.size)}; the limit is ${maxMb} MB.`
  return null
}

export function FileDrop({ label, hint, file, onChange, accept, maxMb, optional }: FileDropProps) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [preview, setPreview] = useState<string | null>(null)

  useEffect(() => {
    if (!file || !PREVIEWABLE.test(file.name)) {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const accept_ = (candidate: File | undefined) => {
    if (!candidate) return
    const problem = validateFile(candidate, accept, maxMb)
    setError(problem)
    onChange(problem ? null : candidate)
  }

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <label htmlFor={inputId} className="text-xs font-medium uppercase tracking-wider text-ink-400">
          {label} {optional && <span className="normal-case text-ink-500">(optional)</span>}
        </label>
        {file && (
          <button
            type="button"
            onClick={() => {
              onChange(null)
              setError(null)
              if (inputRef.current) inputRef.current.value = ''
            }}
            className="flex items-center gap-1 text-xs text-ink-400 hover:text-bad"
            aria-label={`Remove ${label}`}
          >
            <X className="h-3.5 w-3.5" aria-hidden /> Remove
          </button>
        )}
      </div>
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          accept_(e.dataTransfer.files[0])
        }}
        className={cx(
          'relative flex aspect-[4/3] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border-2 border-dashed text-center transition-colors',
          dragging ? 'border-accent-400 bg-accent-500/10' : error ? 'border-bad/60 bg-bad/5' : 'border-ink-600 bg-ink-900/60 hover:border-ink-500',
        )}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={accept.join(',')}
          className="sr-only"
          onChange={(e) => accept_(e.target.files?.[0])}
        />
        {preview ? (
          <img src={preview} alt={`${label} preview`} className="h-full w-full object-cover" />
        ) : file ? (
          <div className="px-4">
            <ImagePlus className="mx-auto h-7 w-7 text-accent-400" aria-hidden />
            <p className="mt-2 text-sm font-medium text-ink-100">GeoTIFF selected</p>
            <p className="text-xs text-ink-400">Preview is generated after upload.</p>
          </div>
        ) : (
          <div className="px-4">
            <ImagePlus className="mx-auto h-7 w-7 text-ink-500" aria-hidden />
            <p className="mt-2 text-sm text-ink-200">Drop an image or click to browse</p>
            <p className="mt-0.5 text-xs text-ink-500">{hint ?? `PNG, JPEG or GeoTIFF · up to ${maxMb} MB`}</p>
          </div>
        )}
        {file && (
          <div className="absolute inset-x-0 bottom-0 truncate bg-ink-950/85 px-3 py-1.5 text-left text-xs text-ink-200">
            {file.name} · {bytes(file.size)}
          </div>
        )}
      </div>
      {error && (
        <p role="alert" className="mt-1.5 text-xs text-bad">
          {error}
        </p>
      )}
    </div>
  )
}
