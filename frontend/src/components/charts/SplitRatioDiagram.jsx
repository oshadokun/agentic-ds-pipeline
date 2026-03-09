/**
 * SplitRatioDiagram
 * Live interactive diagram. Shows train / validation / test as a segmented
 * bar with sliders. Updates in real time as the user drags. Communicates
 * the "test set is sealed" rule visually.
 *
 * Props:
 *   totalRows  — total number of rows in the dataset
 *   onChange   — fn({ train, val, test }) — called when ratios change
 *   initialTrain — starting train % (default 75)
 *   initialVal   — starting val %   (default 10)
 *   disabled     — true after user confirms (freezes sliders)
 */

import { useState, useMemo, useEffect } from "react"
import { CHART_COLOURS } from "./chartTheme"

function SliderRow({ label, value, min, max, colour, onChange, hint, disabled }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1.5">
        <span className="text-gray-700 font-medium">{label}</span>
        <span className="font-mono text-gray-600">{value}%</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={e => !disabled && onChange(Number(e.target.value))}
        disabled={disabled}
        className="w-full h-2 rounded-full appearance-none cursor-pointer
                   disabled:opacity-50 disabled:cursor-default"
        style={{ accentColor: colour }}
        aria-label={`${label} percentage`}
      />
      {hint && (
        <p className="text-xs text-gray-400 mt-1">{hint}</p>
      )}
    </div>
  )
}

export default function SplitRatioDiagram({
  totalRows = 0,
  onChange,
  initialTrain = 75,
  initialVal   = 10,
  disabled     = false
}) {
  const [trainPct, setTrainPct] = useState(initialTrain)
  const [valPct,   setValPct]   = useState(initialVal)
  const testPct = 100 - trainPct - valPct

  const counts = useMemo(() => ({
    train: Math.round(totalRows * trainPct / 100),
    val:   Math.round(totalRows * valPct   / 100),
    test:  Math.round(totalRows * testPct  / 100)
  }), [totalRows, trainPct, valPct, testPct])

  // Notify parent whenever ratios change
  useEffect(() => {
    onChange?.({ train: trainPct / 100, val: valPct / 100, test: testPct / 100 })
  }, [trainPct, valPct])   // eslint-disable-line

  function handleTrainChange(v) {
    if (v + valPct <= 95 && v >= 50) setTrainPct(v)
  }

  function handleValChange(v) {
    if (trainPct + v <= 95 && v >= 5) setValPct(v)
  }

  const segments = [
    {
      label:  "Training",
      pct:    trainPct,
      count:  counts.train,
      colour: CHART_COLOURS.primary,
      hint:   "The model learns from these rows"
    },
    {
      label:  "Validation",
      pct:    valPct,
      count:  counts.val,
      colour: CHART_COLOURS.accent,
      hint:   "Used to tune and check the model during training"
    },
    {
      label:  "Test",
      pct:    testPct,
      count:  counts.test,
      colour: CHART_COLOURS.muted,
      hint:   null
    }
  ]

  return (
    <div>
      {/* Segmented bar */}
      <div
        className="flex rounded-xl overflow-hidden h-11 mb-4 shadow-sm"
        role="img"
        aria-label={`Data split: ${trainPct}% training, ${valPct}% validation, ${testPct}% test`}
      >
        {segments.map(seg => (
          <div
            key={seg.label}
            className="flex items-center justify-center text-white text-xs
                       font-semibold transition-all duration-300 overflow-hidden
                       select-none"
            style={{ width: `${seg.pct}%`, backgroundColor: seg.colour }}
          >
            {seg.pct >= 12 ? `${seg.pct}%` : ""}
          </div>
        ))}
      </div>

      {/* Legend with row counts */}
      <div className="flex flex-wrap gap-5 mb-6">
        {segments.map(seg => (
          <div key={seg.label} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-sm flex-shrink-0"
              style={{ backgroundColor: seg.colour }}
            />
            <div>
              <p className="text-xs font-semibold text-gray-700">{seg.label}</p>
              <p className="text-xs text-gray-400">
                {seg.count.toLocaleString()} rows
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Sliders */}
      <div className="space-y-5">
        <SliderRow
          label="Training"
          value={trainPct}
          min={50} max={90}
          colour={CHART_COLOURS.primary}
          onChange={handleTrainChange}
          hint="The model learns from these rows"
          disabled={disabled}
        />
        <SliderRow
          label="Validation"
          value={valPct}
          min={5} max={25}
          colour={CHART_COLOURS.accent}
          onChange={handleValChange}
          hint="Used to check and tune the model — never used for training"
          disabled={disabled}
        />
        {/* Test — not adjustable, always the remainder */}
        <div>
          <div className="flex justify-between text-sm mb-1.5">
            <span className="text-gray-500">Test (held back)</span>
            <span className="font-mono text-gray-400">{testPct}%</span>
          </div>
          <div className="h-2 bg-gray-100 rounded-full">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${testPct}%`,
                backgroundColor: CHART_COLOURS.muted,
                opacity: 0.5
              }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Sealed until the very end — evaluated exactly once after tuning.
          </p>
        </div>
      </div>
    </div>
  )
}
