/**
 * DriftSummaryDonut
 * Donut chart summarising how many features have drifted and at what severity.
 * Hover segments for feature counts.
 *
 * Props:
 *   total          — total number of features monitored
 *   drifted        — number with any drift detected
 *   highSeverity   — count with high-severity drift
 *   mediumSeverity — count with medium-severity drift
 */

import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer
} from "recharts"
import { CHART_COLOURS, tooltipStyle } from "./chartTheme"

function DriftTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div style={tooltipStyle}>
      <p className="font-semibold" style={{ color: d.colour }}>{d.name}</p>
      <p className="text-gray-600 text-sm">
        {d.value} feature{d.value !== 1 ? "s" : ""}
      </p>
    </div>
  )
}

export default function DriftSummaryDonut({
  total = 0,
  drifted = 0,
  highSeverity = 0,
  mediumSeverity = 0
}) {
  const noDrift      = total - drifted
  const lowSeverity  = Math.max(0, drifted - highSeverity - mediumSeverity)

  const segments = [
    { name: "No drift",     value: noDrift,        colour: CHART_COLOURS.success },
    { name: "Low drift",    value: lowSeverity,    colour: "#D1FAE5"             },
    { name: "Medium drift", value: mediumSeverity, colour: CHART_COLOURS.accent  },
    { name: "High drift",   value: highSeverity,   colour: CHART_COLOURS.error   }
  ].filter(d => d.value > 0)

  if (!total) {
    return (
      <div className="h-36 flex items-center justify-center text-gray-400 text-sm">
        No monitoring data yet.
      </div>
    )
  }

  return (
    <div className="flex items-center gap-6 flex-wrap">
      {/* Donut */}
      <div className="flex-shrink-0">
        <ResponsiveContainer width={140} height={140}>
          <PieChart>
            <Pie
              data={segments}
              cx="50%"
              cy="50%"
              innerRadius={42}
              outerRadius={62}
              paddingAngle={2}
              dataKey="value"
              animationBegin={0}
              animationDuration={400}
            >
              {segments.map((entry, i) => (
                <Cell key={i} fill={entry.colour} />
              ))}
            </Pie>
            <Tooltip content={<DriftTooltip />} />
          </PieChart>
        </ResponsiveContainer>

        {/* Centre label */}
        <div className="text-center -mt-2">
          <p className="text-xs text-gray-400">
            {drifted}/{total} drifted
          </p>
        </div>
      </div>

      {/* Legend */}
      <div className="space-y-2.5">
        {segments.map(d => (
          <div key={d.name} className="flex items-center gap-2 text-sm min-w-[140px]">
            <div
              className="w-3 h-3 rounded-sm flex-shrink-0"
              style={{ backgroundColor: d.colour }}
            />
            <span className="text-gray-700 flex-1">{d.name}</span>
            <span className="text-gray-400 tabular-nums text-xs">
              {d.value} / {total}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
