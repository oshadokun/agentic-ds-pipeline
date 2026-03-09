/**
 * StaticChart
 * Renders a Python-generated PNG served from the backend session directory.
 * Used for: SHAP beeswarm, ROC curve, residual plots, correlation heatmap,
 * waterfall charts, scaling comparison.
 *
 * Every chart is shown with a mandatory caption. Charts are never rendered
 * without explanation (per ui-interaction SKILL).
 *
 * Props:
 *   sessionId  — active session ID
 *   chartPath  — path as returned by backend (e.g. "reports/eda/roc_curve.png")
 *   altText    — descriptive alt text for screen readers (required)
 *   caption    — plain English caption shown below the chart (required)
 *   expandable — allow click-to-expand (default true)
 */

import { useState, useEffect } from "react"
import { Maximize2, Minimize2 } from "lucide-react"

const BASE_URL = "/api"

export default function StaticChart({
  sessionId, chartPath, altText, caption, expandable = true
}) {
  const [src,      setSrc]      = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(false)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (!sessionId || !chartPath) {
      setLoading(false)
      return
    }

    let objectUrl = null
    setLoading(true)
    setError(false)

    const normalised = chartPath.replace(/\\/g, "/")
    const url = `${BASE_URL}/sessions/${sessionId}/charts?path=${encodeURIComponent(normalised)}`
    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error("Not found")
        return r.blob()
      })
      .then(blob => {
        objectUrl = URL.createObjectURL(blob)
        setSrc(objectUrl)
        setLoading(false)
      })
      .catch(() => {
        setError(true)
        setLoading(false)
      })

    return () => {
      // Revoke the object URL when component unmounts to free memory
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [sessionId, chartPath])

  if (!chartPath) return null

  if (loading) {
    return (
      <div className="w-full aspect-video bg-gray-50 rounded-xl border border-gray-100
                      flex items-center justify-center animate-pulse mb-6">
        <p className="text-sm text-gray-400">Loading chart…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="w-full aspect-video bg-red-50 rounded-xl border border-red-100
                      flex items-center justify-center mb-6">
        <p className="text-sm text-red-400">Chart could not be loaded.</p>
      </div>
    )
  }

  return (
    <figure className="mb-6">
      <div
        className={`
          relative rounded-xl overflow-hidden border border-gray-100
          bg-white shadow-sm transition-all duration-200
          ${expandable ? "cursor-pointer hover:shadow-md" : ""}
        `}
        onClick={() => expandable && setExpanded(e => !e)}
      >
        <img
          src={src}
          alt={altText}
          className="w-full h-auto block"
          loading="lazy"
        />

        {expandable && (
          <button
            aria-label={expanded ? "Collapse chart" : "Expand chart"}
            className="absolute top-3 right-3 p-1.5 bg-white/80 rounded-lg
                       text-gray-400 hover:text-gray-700 transition-colors
                       focus:outline-none focus:ring-2 focus:ring-blue-300"
            onClick={e => { e.stopPropagation(); setExpanded(v => !v) }}
          >
            {expanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
          </button>
        )}
      </div>

      {/* Caption — always shown */}
      {caption && (
        <figcaption className="mt-3 px-1 text-sm text-gray-500 leading-relaxed">
          {caption}
        </figcaption>
      )}

      {/* Expanded overlay */}
      {expanded && (
        <div
          className="fixed inset-0 bg-black/60 flex items-center justify-center
                     p-6 z-50 animate-fade-in"
          onClick={() => setExpanded(false)}
        >
          <div className="max-w-4xl w-full max-h-full overflow-auto rounded-2xl
                          bg-white p-4 shadow-xl animate-scale-in"
               onClick={e => e.stopPropagation()}>
            <img src={src} alt={altText} className="w-full h-auto rounded-xl" />
            {caption && (
              <p className="mt-3 text-sm text-gray-500 text-center leading-relaxed">
                {caption}
              </p>
            )}
          </div>
        </div>
      )}
    </figure>
  )
}
