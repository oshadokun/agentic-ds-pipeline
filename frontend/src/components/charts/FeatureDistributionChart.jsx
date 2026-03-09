/**
 * FeatureDistributionChart
 * Interactive bar chart showing the distribution of a single numeric or
 * categorical feature. Hover bars to see exact counts.
 *
 * Props (numeric):
 *   feature          — column name
 *   bins             — [{ range, count, pct, isOutlier }]
 *   highlightOutliers — colour outlier bins amber (default true)
 *
 * Props (categorical):
 *   feature — column name
 *   values  — [{ label, count, pct }]
 *   type    — "numeric" | "categorical"
 */

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from "recharts"
import {
  CHART_COLOURS, CHART_DEFAULTS, tooltipStyle, axisProps
} from "./chartTheme"

function NumericTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div style={tooltipStyle}>
      <p className="font-semibold text-gray-800">{d.range}</p>
      <p className="text-gray-600">
        {d.count.toLocaleString()} rows
        {d.pct !== undefined && (
          <span className="text-gray-400 ml-1">({d.pct}%)</span>
        )}
      </p>
      {d.isOutlier && (
        <p className="text-amber-600 text-xs mt-1">⚠ Potential outlier range</p>
      )}
    </div>
  )
}

function CategoricalTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div style={tooltipStyle}>
      <p className="font-semibold text-gray-800">{d.label}</p>
      <p className="text-gray-600">
        {d.count.toLocaleString()} rows
        {d.pct !== undefined && (
          <span className="text-gray-400 ml-1">({d.pct}%)</span>
        )}
      </p>
    </div>
  )
}

export default function FeatureDistributionChart({
  feature,
  bins,
  values,
  type = "numeric",
  highlightOutliers = true
}) {
  if (type === "categorical" && values?.length) {
    return (
      <div>
        <p className="text-xs text-gray-400 mb-3">
          Hover a bar to see the count for each category.
        </p>
        <ResponsiveContainer width="100%" height={Math.max(180, values.length * 44)}>
          <BarChart
            data={values}
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
              width={110}
              tick={{ ...axisProps.tick, fontSize: 11 }}
            />
            <Tooltip content={<CategoricalTooltip />} cursor={{ fill: "#F9FAFB" }} />
            <Bar
              dataKey="count"
              radius={[0, 4, 4, 0]}
              animationDuration={CHART_DEFAULTS.animationDuration}
            >
              {values.map((_, i) => (
                <Cell
                  key={i}
                  fill={CHART_COLOURS.series[i % CHART_COLOURS.series.length]}
                  fillOpacity={0.82}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    )
  }

  // Numeric histogram
  const data = bins ?? []
  return (
    <div>
      <p className="text-xs text-gray-400 mb-3">
        Hover a bar to see the exact count for that value range.
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={CHART_DEFAULTS.margin}>
          <CartesianGrid vertical={false} stroke={CHART_COLOURS.grid} />
          <XAxis
            dataKey="range"
            {...axisProps}
            tick={{ ...axisProps.tick, fontSize: 10 }}
            interval="preserveStartEnd"
          />
          <YAxis
            {...axisProps}
            tickFormatter={v => v.toLocaleString()}
            width={48}
          />
          <Tooltip content={<NumericTooltip />} cursor={{ fill: "#F9FAFB" }} />
          <Bar
            dataKey="count"
            radius={[3, 3, 0, 0]}
            animationDuration={CHART_DEFAULTS.animationDuration}
          >
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={
                  highlightOutliers && entry.isOutlier
                    ? CHART_COLOURS.accent
                    : CHART_COLOURS.primary
                }
                fillOpacity={0.82}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
