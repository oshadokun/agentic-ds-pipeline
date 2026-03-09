/**
 * FeatureImportanceChart
 * Interactive horizontal bar chart of SHAP feature importance.
 * Hover for exact value and relative influence.
 *
 * Props:
 *   features — [{ feature, importance }] sorted descending (top 15 shown)
 */

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from "recharts"
import {
  CHART_COLOURS, CHART_DEFAULTS, tooltipStyle, axisProps
} from "./chartTheme"

function ImportanceTooltip({ active, payload, maxVal }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const relative = maxVal > 0
    ? ((d.importance / maxVal) * 100).toFixed(0)
    : 0
  return (
    <div style={tooltipStyle}>
      <p className="font-semibold text-gray-800 mb-1">{d.feature}</p>
      <p className="text-gray-600 text-xs">
        Average influence:{" "}
        <span className="font-mono text-gray-800">{d.importance.toFixed(4)}</span>
      </p>
      <p className="text-blue-600 text-xs mt-0.5">
        {relative}% of the most influential feature
      </p>
    </div>
  )
}

export default function FeatureImportanceChart({ features = [] }) {
  const top    = features.slice(0, 15)
  const maxVal = top[0]?.importance ?? 1

  if (!top.length) {
    return (
      <div className="h-40 flex items-center justify-center text-gray-400 text-sm">
        No feature importance data available.
      </div>
    )
  }

  // Reverse so most important is at the top of the chart
  const data = [...top].reverse()

  return (
    <div>
      <p className="text-xs text-gray-400 mb-3">
        Longer bar = greater average influence on the model's predictions.
        Hover a bar for the exact value.
      </p>
      <ResponsiveContainer width="100%" height={Math.max(200, top.length * 32)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 56, bottom: 4, left: 8 }}
        >
          <CartesianGrid horizontal={false} stroke={CHART_COLOURS.grid} />
          <XAxis
            type="number"
            tickFormatter={v => v.toFixed(3)}
            {...axisProps}
          />
          <YAxis
            type="category"
            dataKey="feature"
            width={130}
            {...axisProps}
            tick={{ ...axisProps.tick, fontSize: 12 }}
          />
          <Tooltip
            content={<ImportanceTooltip maxVal={maxVal} />}
            cursor={{ fill: "#F9FAFB" }}
          />
          <Bar
            dataKey="importance"
            radius={[0, 4, 4, 0]}
            label={{
              position: "right",
              formatter: v => v.toFixed(3),
              fontSize: 11,
              fill: "#9CA3AF"
            }}
            animationDuration={CHART_DEFAULTS.animationDuration}
          >
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={CHART_COLOURS.primary}
                fillOpacity={0.75 + (0.25 * (1 - i / data.length))}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
