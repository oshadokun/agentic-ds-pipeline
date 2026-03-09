/**
 * InteractiveConfusionMatrix
 * Hover cells for plain English breakdown of what each cell means.
 * Works for binary and multiclass classification.
 *
 * Props:
 *   matrix     — 2D array, e.g. [[TN, FP], [FN, TP]]
 *   classNames — array of class label strings
 */

import { useState } from "react"

function cellColour(row, col, value, maxVal) {
  const intensity = maxVal > 0 ? value / maxVal : 0
  const isCorrect = row === col
  // Correct cells → green tint; incorrect → red tint
  const [r, g, b] = isCorrect ? [6, 95, 70] : [153, 27, 27]
  return `rgba(${r},${g},${b},${0.07 + intensity * 0.72})`
}

function binaryLabel(row, col) {
  if (row === 0 && col === 0) return "True Negative"
  if (row === 0 && col === 1) return "False Positive"
  if (row === 1 && col === 0) return "False Negative"
  if (row === 1 && col === 1) return "True Positive"
}

function binaryDescription(row, col, count, total, classNames) {
  const pct   = total > 0 ? ((count / total) * 100).toFixed(1) : 0
  const cn    = classNames
  const descs = {
    "00": `${count.toLocaleString()} rows (${pct}%) were correctly predicted as '${cn[0]}'`,
    "01": `${count.toLocaleString()} rows (${pct}%) were predicted as '${cn[1]}' but were actually '${cn[0]}' — false alarms`,
    "10": `${count.toLocaleString()} rows (${pct}%) were predicted as '${cn[0]}' but were actually '${cn[1]}' — missed cases`,
    "11": `${count.toLocaleString()} rows (${pct}%) were correctly predicted as '${cn[1]}'`
  }
  return descs[`${row}${col}`] ?? ""
}

function genericDescription(row, col, count, total, classNames) {
  const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0
  if (row === col) {
    return `${count.toLocaleString()} rows (${pct}%) correctly predicted as '${classNames[row]}'`
  }
  return `${count.toLocaleString()} rows (${pct}%) were predicted as '${classNames[col]}' but were actually '${classNames[row]}'`
}

export default function InteractiveConfusionMatrix({ matrix = [], classNames = [] }) {
  const [hovered, setHovered] = useState(null)

  if (!matrix.length || !classNames.length) return null

  const total  = matrix.flat().reduce((a, b) => a + b, 0)
  const maxVal = Math.max(...matrix.flat())
  const isBinary = classNames.length === 2

  function getDescription(ri, ci, count) {
    return isBinary
      ? binaryDescription(ri, ci, count, total, classNames)
      : genericDescription(ri, ci, count, total, classNames)
  }

  return (
    <div>
      {/* Column headers — Predicted */}
      <div className="mb-1">
        <p className="text-xs text-gray-400 text-center mb-1.5 font-medium">
          Predicted →
        </p>
        <div
          className="grid"
          style={{ gridTemplateColumns: `72px repeat(${classNames.length}, 1fr)` }}
        >
          <div />
          {classNames.map(name => (
            <div
              key={name}
              className="text-center text-xs font-semibold text-gray-600 py-1 px-1 truncate"
            >
              {name}
            </div>
          ))}
        </div>
      </div>

      {/* Row label indicator */}
      <div className="flex">
        <div className="flex items-center justify-center" style={{ width: 72 }}>
          <span
            className="text-xs text-gray-400 font-medium select-none"
            style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
          >
            Actual ↑
          </span>
        </div>

        {/* Matrix grid */}
        <div className="flex-1">
          {matrix.map((row, ri) => (
            <div
              key={ri}
              className="grid mb-1"
              style={{ gridTemplateColumns: `repeat(${classNames.length}, 1fr)` }}
            >
              {row.map((count, ci) => (
                <button
                  key={ci}
                  onMouseEnter={() => setHovered({ ri, ci, count })}
                  onMouseLeave={() => setHovered(null)}
                  onFocus={() => setHovered({ ri, ci, count })}
                  onBlur={() => setHovered(null)}
                  className="relative aspect-square rounded-lg m-0.5 flex flex-col
                             items-center justify-center transition-all duration-150
                             focus:outline-none focus:ring-2 focus:ring-blue-400
                             hover:scale-105"
                  style={{ backgroundColor: cellColour(ri, ci, count, maxVal) }}
                  aria-label={getDescription(ri, ci, count)}
                >
                  <span className="text-base font-bold text-gray-800 leading-none">
                    {count.toLocaleString()}
                  </span>
                  {isBinary && (
                    <span className="text-[10px] text-gray-500 mt-0.5 leading-tight text-center px-1">
                      {binaryLabel(ri, ci)}
                    </span>
                  )}
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Row labels — Actual */}
      <div
        className="grid mt-1 ml-[72px]"
        style={{ gridTemplateColumns: `repeat(${classNames.length}, 1fr)` }}
      >
        {classNames.map((name, i) => (
          <div
            key={i}
            className="text-center text-xs font-semibold text-gray-500 py-1 px-1 truncate"
          >
            {name}
          </div>
        ))}
      </div>

      {/* Hover description */}
      <div
        aria-live="polite"
        className={`
          mt-4 p-3 rounded-xl text-sm transition-all duration-200 min-h-[48px]
          ${hovered
            ? "bg-gray-50 border border-gray-200 text-gray-700"
            : "text-gray-300 text-center"}
        `}
      >
        {hovered
          ? getDescription(hovered.ri, hovered.ci, hovered.count)
          : "Hover a cell to understand what it means"}
      </div>
    </div>
  )
}
