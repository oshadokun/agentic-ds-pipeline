/**
 * TuningTrialChart
 * Line chart of Optuna trial scores over time.
 * Shows individual trial scores (grey dots) and the running best (blue step line).
 * Baseline score shown as a dashed reference line.
 *
 * Props:
 *   trials        — [{ trial, score }]
 *   metricName    — e.g. "ROC-AUC"
 *   baselineScore — score before tuning
 */

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer
} from "recharts"
import {
  CHART_COLOURS, CHART_DEFAULTS, tooltipStyle, axisProps
} from "./chartTheme"

function TrialTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const trial      = payload.find(p => p.dataKey === "score")
  const runningBest = payload.find(p => p.dataKey === "runningBest")
  return (
    <div style={tooltipStyle}>
      <p className="font-semibold text-gray-700 mb-1">Trial {label}</p>
      {trial && (
        <p className="text-gray-500 text-xs">
          Score: <span className="font-mono text-gray-700">{trial.value?.toFixed(4)}</span>
        </p>
      )}
      {runningBest && (
        <p className="text-blue-600 text-xs">
          Best so far: <span className="font-mono font-semibold">{runningBest.value?.toFixed(4)}</span>
        </p>
      )}
    </div>
  )
}

export default function TuningTrialChart({ trials = [], metricName = "Score", baselineScore }) {
  // Compute running best
  const data = trials.reduce((acc, t, i) => {
    const prev        = acc[i - 1]?.runningBest ?? 0
    const runningBest = Math.max(prev, t.score ?? 0)
    return [...acc, { ...t, runningBest }]
  }, [])

  if (!data.length) {
    return (
      <div className="h-40 flex items-center justify-center text-gray-400 text-sm">
        No trial data available.
      </div>
    )
  }

  return (
    <div>
      <p className="text-xs text-gray-400 mb-3">
        Each dot is one combination of settings tried. The blue line shows the best
        score found so far. Hover for details.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ ...CHART_DEFAULTS.margin, left: 8 }}>
          <CartesianGrid stroke={CHART_COLOURS.grid} strokeDasharray="3 3" />
          <XAxis
            dataKey="trial"
            {...axisProps}
            label={{
              value: "Trial number",
              position: "insideBottom",
              offset: -4,
              fontSize: 11,
              fill: "#9CA3AF"
            }}
          />
          <YAxis
            domain={["auto", "auto"]}
            tickFormatter={v => v.toFixed(3)}
            {...axisProps}
            width={52}
            label={{
              value: metricName,
              angle: -90,
              position: "insideLeft",
              fontSize: 11,
              fill: "#9CA3AF",
              dx: -4
            }}
          />
          <Tooltip content={<TrialTooltip />} />

          {/* Baseline reference */}
          {baselineScore !== undefined && (
            <ReferenceLine
              y={baselineScore}
              stroke={CHART_COLOURS.muted}
              strokeDasharray="5 3"
              label={{
                value: "Baseline",
                position: "insideTopRight",
                fontSize: 11,
                fill: "#9CA3AF"
              }}
            />
          )}

          {/* Individual trial scores — grey dots */}
          <Line
            type="monotone"
            dataKey="score"
            stroke={CHART_COLOURS.muted}
            dot={{ r: 2, fill: CHART_COLOURS.muted, strokeWidth: 0 }}
            strokeWidth={1}
            isAnimationActive={false}
            name="Trial score"
          />

          {/* Running best — blue step line */}
          <Line
            type="stepAfter"
            dataKey="runningBest"
            stroke={CHART_COLOURS.primary}
            dot={false}
            strokeWidth={CHART_DEFAULTS.strokeWidth}
            animationDuration={CHART_DEFAULTS.animationDuration}
            name="Best so far"
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center gap-5 mt-3 pl-1">
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-0.5 bg-[#9CA3AF]" />
          <span className="text-xs text-gray-400">Trial score</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-0.5 bg-[#1B3A5C]" />
          <span className="text-xs text-gray-600">Best so far</span>
        </div>
        {baselineScore !== undefined && (
          <div className="flex items-center gap-1.5">
            <div className="w-4 border-t border-dashed border-[#9CA3AF]" />
            <span className="text-xs text-gray-400">Baseline</span>
          </div>
        )}
      </div>
    </div>
  )
}
