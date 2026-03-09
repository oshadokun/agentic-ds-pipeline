/**
 * TargetDistributionChart
 * Shows the distribution of the target column.
 *
 * For classification: horizontal bar chart of class counts with imbalance warning.
 * For regression:     delegates to FeatureDistributionChart (histogram).
 *
 * Props:
 *   targetCol — column name
 *   taskType  — "binary_classification" | "multiclass_classification" | "regression"
 *   data      — classification: [{ label, count, pct }]
 *               regression:     [{ range, count, pct, isOutlier }]
 */

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from "recharts"
import {
  CHART_COLOURS, CHART_DEFAULTS, tooltipStyle, axisProps
} from "./chartTheme"
import FeatureDistributionChart from "./FeatureDistributionChart"

function ClassTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div style={tooltipStyle}>
      <p className="font-semibold text-gray-800">{d.label}</p>
      <p className="text-gray-600">
        {d.count.toLocaleString()} rows — {d.pct}% of dataset
      </p>
      {d.pct < 15 && (
        <p className="text-amber-600 text-xs mt-1">
          ⚠ Minority class — imbalance detected
        </p>
      )}
    </div>
  )
}

export default function TargetDistributionChart({ targetCol, taskType, data = [] }) {
  if (taskType === "regression") {
    return (
      <FeatureDistributionChart
        feature={targetCol}
        bins={data}
        type="numeric"
      />
    )
  }

  return (
    <div>
      <p className="text-xs text-gray-400 mb-3">
        Hover a bar to see the count for each outcome.
        {data.some(d => d.pct < 15) && (
          <span className="text-amber-600 ml-1">
            ⚠ Imbalanced classes detected.
          </span>
        )}
      </p>
      <ResponsiveContainer
        width="100%"
        height={Math.max(120, data.length * 52)}
      >
        <BarChart
          data={data}
          layout="vertical"
          margin={{ ...CHART_DEFAULTS.margin, left: 8 }}
        >
          <CartesianGrid horizontal={false} stroke={CHART_COLOURS.grid} />
          <XAxis
            type="number"
            {...axisProps}
            tickFormatter={v => v.toLocaleString()}
          />
          <YAxis
            type="category"
            dataKey="label"
            {...axisProps}
            width={90}
            tick={{ ...axisProps.tick, fontSize: 12 }}
          />
          <Tooltip content={<ClassTooltip />} cursor={{ fill: "#F9FAFB" }} />
          <Bar
            dataKey="count"
            radius={[0, 4, 4, 0]}
            animationDuration={CHART_DEFAULTS.animationDuration}
          >
            {data.map((_, i) => (
              <Cell
                key={i}
                fill={CHART_COLOURS.series[i % CHART_COLOURS.series.length]}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
