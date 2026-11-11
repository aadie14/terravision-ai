import { useEffect, useState } from 'react'

export const CHANGE_RGB: [number, number, number] = [255, 59, 92]

/** Convert a white-on-black mask image into a coloured RGBA data URL (transparent where no change). */
export function colorizeMask(image: CanvasImageSource, width: number, height: number, rgb = CHANGE_RGB): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) throw new Error('2D canvas is not available')
  ctx.drawImage(image, 0, 0, width, height)
  const data = ctx.getImageData(0, 0, width, height)
  const px = data.data
  for (let i = 0; i < px.length; i += 4) {
    const on = px[i] > 127
    px[i] = rgb[0]
    px[i + 1] = rgb[1]
    px[i + 2] = rgb[2]
    px[i + 3] = on ? 255 : 0
  }
  ctx.putImageData(data, 0, 0)
  return canvas.toDataURL('image/png')
}

/** Load the mask artefact and return a tinted transparent overlay URL (null while loading). */
export function useMaskOverlay(maskUrl: string | null): { overlay: string | null; failed: boolean } {
  const [overlay, setOverlay] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    setOverlay(null)
    setFailed(false)
    if (!maskUrl) return
    let cancelled = false
    const img = new Image()
    img.onload = () => {
      if (cancelled) return
      try {
        setOverlay(colorizeMask(img, img.naturalWidth, img.naturalHeight))
      } catch {
        setFailed(true)
      }
    }
    img.onerror = () => {
      if (!cancelled) setFailed(true)
    }
    img.src = maskUrl
    return () => {
      cancelled = true
    }
  }, [maskUrl])
  return { overlay, failed }
}
