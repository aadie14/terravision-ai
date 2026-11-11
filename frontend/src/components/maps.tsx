import L from 'leaflet'
import { useEffect, useRef } from 'react'
import { ImageOverlay, MapContainer, Rectangle, TileLayer, useMap, useMapEvents } from 'react-leaflet'

import type { BBox, Bounds } from '../lib/types'

const OSM_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

function toLatLngBounds(b: Bounds | BBox): L.LatLngBoundsExpression {
  return [
    [b.south, b.west],
    [b.north, b.east],
  ]
}

/** Click-drag rectangle drawing. While `active`, map panning is suspended. */
function RectangleDrawer({ active, onDrawn }: { active: boolean; onDrawn: (bbox: BBox) => void }) {
  const map = useMap()
  const start = useRef<L.LatLng | null>(null)
  const rubber = useRef<L.Rectangle | null>(null)

  useEffect(() => {
    const container = map.getContainer()
    if (active) {
      map.dragging.disable()
      container.classList.add('drawing')
    } else {
      map.dragging.enable()
      container.classList.remove('drawing')
    }
    return () => {
      map.dragging.enable()
      container.classList.remove('drawing')
    }
  }, [active, map])

  useMapEvents({
    mousedown(e) {
      if (!active) return
      start.current = e.latlng
      rubber.current = L.rectangle(L.latLngBounds(e.latlng, e.latlng), { color: '#2dd4bf', weight: 2, dashArray: '6 4' }).addTo(map)
    },
    mousemove(e) {
      if (!active || !start.current || !rubber.current) return
      rubber.current.setBounds(L.latLngBounds(start.current, e.latlng))
    },
    mouseup(e) {
      if (!active || !start.current) return
      const bounds = L.latLngBounds(start.current, e.latlng)
      rubber.current?.remove()
      rubber.current = null
      start.current = null
      if (bounds.getNorth() - bounds.getSouth() < 1e-5 || bounds.getEast() - bounds.getWest() < 1e-5) return
      onDrawn({ west: bounds.getWest(), south: bounds.getSouth(), east: bounds.getEast(), north: bounds.getNorth() })
    },
  })
  return null
}

function FlyTo({ target }: { target: Bounds | BBox | null }) {
  const map = useMap()
  useEffect(() => {
    if (target) map.fitBounds(toLatLngBounds(target), { padding: [30, 30], maxZoom: 15 })
  }, [target, map])
  return null
}

export interface OverlayLayer {
  key: string
  url: string
  bounds: Bounds
  opacity: number
}

/** Leaflet map with an optional AOI rectangle, drawing mode and georeferenced image overlays. */
export function GeoMap({ aoi, drawing = false, onDrawn, focus, overlays = [], height = 520 }: {
  aoi?: BBox | null
  drawing?: boolean
  onDrawn?: (bbox: BBox) => void
  focus?: Bounds | BBox | null
  overlays?: OverlayLayer[]
  height?: number
}) {
  return (
    <MapContainer
      center={[13.115, 77.63]}
      zoom={11}
      minZoom={2}
      scrollWheelZoom
      style={{ height, width: '100%' }}
      worldCopyJump
    >
      <TileLayer url={OSM_URL} attribution={OSM_ATTRIBUTION} maxZoom={19} />
      {overlays.map((layer) => (
        <ImageOverlay key={layer.key} url={layer.url} bounds={toLatLngBounds(layer.bounds)} opacity={layer.opacity} className="pixelated" />
      ))}
      {aoi && <Rectangle bounds={toLatLngBounds(aoi)} pathOptions={{ color: '#2dd4bf', weight: 2, fillOpacity: 0.06 }} />}
      {onDrawn && <RectangleDrawer active={drawing} onDrawn={onDrawn} />}
      <FlyTo target={focus ?? null} />
    </MapContainer>
  )
}

/** Approximate box size in km (equirectangular; good enough for an on-screen readout). */
// eslint-disable-next-line react-refresh/only-export-components
export function bboxSizeKm(b: BBox): { width: number; height: number } {
  const midLat = ((b.south + b.north) / 2) * (Math.PI / 180)
  return { width: (b.east - b.west) * 111.32 * Math.cos(midLat), height: (b.north - b.south) * 110.574 }
}
