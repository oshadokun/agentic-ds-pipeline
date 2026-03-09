/**
 * PerformanceTrendChart
 * Time-series line chart of model performance across monitoring reports.
 * Shows deployment baseline, warning zone, and the trend over time.
 *
 * Props:
 *   reports            — [{ reportNumber, date, score }]
 *   metricName         — e.g. "ROC-AUC"
 *   baselineScore      — score at deployment
 *   warningThreshold   — score below which to shade warning zone (optional)
 *   criticalThreshold  — lower bound of warning zone (optional)
 */

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ReferenceArea, ResponsiveContainer
} from "recharts"
import {
  CHART_COLOURS, CHART_DEFAULTS, tooltipStyle, axisProps
} from "./chartTheme"

function TrendTooltip({ active, payload, label, baselineScore, metricName }) {
  if (!active || !payload?.length) return null
  const score  = payload[0]?.value
  const change = score !== undefined && baselineScore !== undefined
    ? score - baselineScore
    : null
  const d = payload[0]?.payload
  return (
    <div style={tooltipStyle}>
      <p className="font-semibold text-gray-700">Report {label}</p>
      {d?.date && (
        <p className="text-gray-400 text-xs mb-1">{d.date}</p>
      )}
      <p className="text-gray-800">
        {metricName}:{" "}
        <span className="font-mono font-semibold">{score?.toFixed(4)}</span>
      </p>
      {change !== null && (
        <p className={`text-xs mt-0.5 font-medium ${change >= 0 ? "text-green-600" : "text-red-600"}`}>
          {change >= 0 ? "+" : ""}{(change * 100).toFixed(1)}% vs deployment
        </p>
      )}
    </div>
  )
}

export default function PerformanceTrendChart({
  reports = [],
  metricName = "Score",
  baselineScore,
  warningThreshold,
  criticalThreshold
}) {
  if (!reports.length) {
    return (
      <div className="h-40 flex items-center justify-center text-gray-400 text-sm">
        No monitoring reports yet.
      </div>
    )
  }

  return (
    <div>
      <p className="text-xs text-gray-400 mb-3">
        Each point is one monitoring check. The dashed line is the score at
        deployment. Hover for details.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart
          data={reports}
          margin={{ ...CHART_DEFAULTS.margin, left: 8 }}
        >
          <CartesianGrid stroke={CHART_COLOURS.grid} strokeDasharray="3 3" />
          <XAxis
            dataKey="reportNumber"
            {...axisProps}
            label={{
              value: "Report number",
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
          <Tooltip
            content={
              <TrendTooltip
                baselineScore={baselineScore}
                metricName={metricName}
              />
            }
          />

          {/* Warning zone shading */}
          {warningThreshold !== undefined && criticalThreshold !== undefined && (
            <ReferenceArea
              y1={criticalThreshold}
              y2={warningThreshold}
              fill={CHART_COLOURS.warning}
              fillOpacity={0.07}
            />
          )}

          {/* Baseline deployment score */}
          {baselineScore !== undefined && (
            <ReferenceLine
              y={baselineScore}
              stroke={CHART_COLOURS.primary}
              strokeDasharray="5 3"
              strokeOpacity={0.6}
              label={{
                value: "Deployment",
                position: "insideTopLeft",
                fontSize: 11,
                fill: CHART_COLOURS.primary
              }}
            />
          )}

          <Line
            type="monotone"
            dataKey="score"
            stroke={CHART_COLOURS.primary}
            strokeWidth={CHART_DEFAULTS.strokeWidth}
            dot={{
              r:           CHART_DEFAULTS.dotRadius,
              fill:        CHART_COLOURS.primary,
              strokeWidth: 2,
              stroke:      "#fff"
            }}
            activeDot={{ r: 6 }}
            animationDuration={CHART_DEFAULTS.animationDuration}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
