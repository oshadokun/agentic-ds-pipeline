---
name: ui-charts
description: >
  Defines all interactive chart components used across the pipeline UI. Always
  read alongside ui-shell and ui-interaction when building any view that contains
  data visualisation. Covers which charts are interactive (Recharts) versus static
  (PNG served from backend), the exact component for each stage, how data flows
  from the backend result JSON into each chart, and the shared chart design system.
  Interactive charts are used where the user benefits from hover, zoom, or live
  updates. Static PNGs are used for complex outputs that Python generates better
  (SHAP plots, seaborn heatmaps). Trigger whenever building or modifying any
  chart, graph, plot, or data visualisation in the pipeline frontend.
---

# UI Charts Skill

Two types of visualisation exist in the pipeline UI. Choosing the right one
for each situation is as important as choosing the right chart type.

**Interactive (Recharts):** Built in the browser from JSON data returned by
the backend. Supports hover tooltips, responsive sizing, and live updates.
Use when the data is simple enough to transmit as JSON and the user benefits
from exploration.

**Static (PNG from backend):** Python-generated images served from the session
reports directory. Use for complex outputs — SHAP beeswarm plots, seaborn
heatmaps, matplotlib residual plots — where Python's output is richer than
what Recharts can produce feasibly.

---

## Chart Assignment by Stage

| Stage | Chart | Type | Why |
|---|---|---|---|
| EDA | Feature distributions | Interactive | User can hover bars to see exact counts |
| EDA | Target distribution | Interactive | Simple bar/pie — hover adds value |
| EDA | Correlation heatmap | Static PNG | Complex colour matrix — Python does it better |
| EDA | Feature vs target | Static PNG | Scatter/box plots — Python does it better |
| Splitting | Split ratio diagram | Interactive | Live — updates as user moves the slider |
| Training | Baseline vs model comparison | Interactive | Simple bar — hover shows exact metric |
| Evaluation | Confusion matrix | Interactive | Hover cells for counts and percentages |
| Evaluation | ROC curve | Static PNG | Precise curve — Python renders it better |
| Evaluation | Residual plot | Static PNG | Scatter density — Python does it better |
| Tuning | Trial score history | Interactive | Line chart of Optuna trial progress |
| Tuning | Before/after comparison | Interactive | Simple bar pair — hover adds value |
| Explainability | Feature importance bar | Interactive | Hover for exact SHAP value |
| Explainability | SHAP summary (beeswarm) | Static PNG | Dense scatter — Python only |
| Explainability | Waterfall (individual) | Static PNG | Complex layout — Python only |
| Monitoring | Drift severity summary | Interactive | Donut chart — hover for feature count |
| Monitoring | Performance trend | Interactive | Line chart over time — essential for trends |
| Monitoring | Feature drift table | Interactive | Sortable table with inline status badges |

---

## Shared Chart Design System

All Recharts components use these shared defaults.

```javascript
// chartTheme.js

export const CHART_COLOURS = {
  primary:   "#1B3A5C",
  secondary: "#2E6099",
  accent:    "#D97706",
  success:   "#065F46",
  warning:   "#92400E",
  error:     "#991B1B",
  muted:     "#9CA3AF",
  grid:      "#F3F4F6",
  // Sequential palette for multi-series
  series: ["#1B3A5C", "#2E6099", "#D97706", "#065F46", "#7C3AED"]
}

export const CHART_DEFAULTS = {
  margin:      { top: 16, right: 16, bottom: 16, left: 16 },
  fontSize:    13,
  fontFamily:  "'DM Sans', system-ui, sans-serif",
  strokeWidth: 2,
  dotRadius:   4,
  animationDuration: 400
}

// Shared tooltip style
export const tooltipStyle = {
  backgroundColor: "#fff",
  border:          "1px solid #E5E7EB",
  borderRadius:    "10px",
  boxShadow:       "0 4px 16px rgba(0,0,0,0.08)",
  fontSize:        "13px",
  fontFamily:      "'DM Sans', system-ui, sans-serif",
  padding:         "8px 12px"
}

// Shared axis props
export const axisProps = {
  tick:     { fontSize: 12, fill: "#6B7280" },
  axisLine: { stroke: "#E5E7EB" },
  tickLine: { stroke: "#E5E7EB" }
}
```

---

## 1. Feature Distribution Chart (EDA)

Shown for each numeric feature. Bar chart of value bins with hover counts.
Replaces the static Python histogram for numeric columns.

```jsx
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from "recharts"
import { CHART_COLOURS, CHART_DEFAULTS, tooltipStyle, axisProps } from "./chartTheme"

function FeatureDistributionChart({ feature, bins, highlightOutliers = false }) {
  // bins: [{ range: "10–20", count: 142, isOutlier: false }, ...]

  const CustomTooltip = ({ active, payload }) => {
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

  return (
    <div>
      <p className="text-xs text-gray-500 mb-3">
        Hover a bar to see the exact count for that range.
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={bins} margin={CHART_DEFAULTS.margin}>
          <CartesianGrid vertical={false} stroke={CHART_COLOURS.grid} />
          <XAxis
            dataKey="range"
            {...axisProps}
            tick={{ ...axisProps.tick, fontSize: 11 }}
            interval="preserveStartEnd"
          />
          <YAxis
            {...axisProps}
            tickFormatter={v => v.toLocaleString()}
            width={48}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "#F3F4F6" }} />
          <Bar
            dataKey="count"
            radius={[4, 4, 0, 0]}
            animationDuration={CHART_DEFAULTS.animationDuration}
          >
            {bins.map((entry, i) => (
              <Cell
                key={i}
                fill={
                  highlightOutliers && entry.isOutlier
                    ? CHART_COLOURS.warning
                    : CHART_COLOURS.primary
                }
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
```

**Data format from backend:**
```json
{
  "feature": "age",
  "bins": [
    { "range": "18–25", "count": 312, "pct": 8.2, "isOutlier": false },
    { "range": "25–35", "count": 891, "pct": 23.4, "isOutlier": false }
  ]
}
```

---

## 2. Target Distribution Chart (EDA)

For classification: horizontal bar chart of class counts.
For regression: histogram. Both interactive.

```jsx
function TargetDistributionChart({ targetCol, taskType, data }) {
  // Classification: data = [{ label: "Churned", count: 412, pct: 18.3 }, ...]
  // Regression:     data = bins array (same as FeatureDistributionChart)

  if (taskType === "regression") {
    return <FeatureDistributionChart feature={targetCol} bins={data} />
  }

  const CustomTooltip = ({ active, payload }) => {
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

  return (
    <ResponsiveContainer width="100%" height={Math.max(120, data.length * 52)}>
      <BarChart data={data} layout="vertical" margin={CHART_DEFAULTS.margin}>
        <CartesianGrid horizontal={false} stroke={CHART_COLOURS.grid} />
        <XAxis type="number" {...axisProps}
               tickFormatter={v => v.toLocaleString()} />
        <YAxis type="category" dataKey="label" {...axisProps} width={90} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "#F3F4F6" }} />
        <Bar dataKey="count" radius={[0, 4, 4, 0]}
             animationDuration={CHART_DEFAULTS.animationDuration}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={CHART_COLOURS.series[i % CHART_COLOURS.series.length]}
              fillOpacity={0.85}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
```

---

## 3. Split Ratio Diagram (Splitting)

Live interactive diagram. Updates in real time as the user adjusts sliders.
Shows train / validation / test proportions as a segmented bar plus row counts.

```jsx
import { useState, useMemo } from "react"

function SplitRatioDiagram({ totalRows, onChange }) {
  const [trainPct, setTrainPct] = useState(75)
  const [valPct,   setValPct]   = useState(10)
  const testPct = 100 - trainPct - valPct

  const counts = useMemo(() => ({
    train: Math.round(totalRows * trainPct / 100),
    val:   Math.round(totalRows * valPct   / 100),
    test:  Math.round(totalRows * testPct  / 100)
  }), [totalRows, trainPct, valPct, testPct])

  function handleTrainChange(v) {
    const newTrain = Number(v)
    // Keep val fixed, adjust test
    if (newTrain + valPct <= 95) setTrainPct(newTrain)
  }

  function handleValChange(v) {
    const newVal = Number(v)
    if (trainPct + newVal <= 95) setValPct(newVal)
  }

  // Notify parent of changes
  useMemo(() => {
    onChange?.({ train: trainPct/100, val: valPct/100, test: testPct/100 })
  }, [trainPct, valPct])

  const segments = [
    { label: "Training",   pct: trainPct, count: counts.train, colour: CHART_COLOURS.primary },
    { label: "Validation", pct: valPct,   count: counts.val,   colour: CHART_COLOURS.accent },
    { label: "Test",       pct: testPct,  count: counts.test,  colour: CHART_COLOURS.muted }
  ]

  return (
    <div>
      {/* Segmented bar */}
      <div className="flex rounded-xl overflow-hidden h-12 mb-3 shadow-sm">
        {segments.map(seg => (
          <div
            key={seg.label}
            className="flex items-center justify-center text-white text-xs
                       font-semibold transition-all duration-300 overflow-hidden"
            style={{ width: `${seg.pct}%`, backgroundColor: seg.colour }}
          >
            {seg.pct >= 12 && `${seg.pct}%`}
          </div>
        ))}
      </div>

      {/* Legend with counts */}
      <div className="flex gap-6 mb-6">
        {segments.map(seg => (
          <div key={seg.label} className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm flex-shrink-0"
                 style={{ backgroundColor: seg.colour }} />
            <div>
              <p className="text-xs font-medium text-gray-700">{seg.label}</p>
              <p className="text-xs text-gray-400">
                {seg.count.toLocaleString()} rows
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Sliders */}
      <div className="space-y-4">
        <SliderRow
          label="Training"
          value={trainPct}
          min={50} max={90}
          colour={CHART_COLOURS.primary}
          onChange={handleTrainChange}
          hint="The model learns from these rows"
        />
        <SliderRow
          label="Validation"
          value={valPct}
          min={5} max={25}
          colour={CHART_COLOURS.accent}
          onChange={handleValChange}
          hint="Used to tune and check the model during training"
        />
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-500">Test</span>
          <span className="text-gray-400 text-xs">
            {testPct}% · {counts.test.toLocaleString()} rows
            <span className="ml-2 text-gray-300">
              (set aside — evaluated only at the very end)
            </span>
          </span>
        </div>
      </div>
    </div>
  )
}

function SliderRow({ label, value, min, max, colour, onChange, hint }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1.5">
        <span className="text-gray-700 font-medium">{label}</span>
        <span className="text-gray-500">{value}%</span>
      </div>
      <input
        type="range"
        min={min} max={max} value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full h-2 rounded-full appearance-none cursor-pointer"
        style={{ accentColor: colour }}
      />
      <p className="text-xs text-gray-400 mt-1">{hint}</p>
    </div>
  )
}
```

---

## 4. Interactive Confusion Matrix (Evaluation)

Hover cells to see plain English breakdown of what each cell means.

```jsx
function InteractiveConfusionMatrix({ matrix, classNames }) {
  const [hovered, setHovered] = useState(null)

  const total = matrix.flat().reduce((a, b) => a + b, 0)
  const maxVal = Math.max(...matrix.flat())

  function cellLabel(row, col) {
    const isBinary = classNames.length === 2
    if (!isBinary) return null
    if (row === 0 && col === 0) return "True Negative"
    if (row === 0 && col === 1) return "False Positive"
    if (row === 1 && col === 0) return "False Negative"
    if (row === 1 && col === 1) return "True Positive"
  }

  function cellDescription(row, col, count) {
    const pct  = ((count / total) * 100).toFixed(1)
    const isBinary = classNames.length === 2
    if (!isBinary) return `${count} cases predicted as '${classNames[col]}', actually '${classNames[row]}'`
    const descriptions = {
      "00": `${count} rows (${pct}%) were correctly predicted as '${classNames[0]}'`,
      "01": `${count} rows (${pct}%) were predicted as '${classNames[1]}' but were actually '${classNames[0]}' — false alarms`,
      "10": `${count} rows (${pct}%) were predicted as '${classNames[0]}' but were actually '${classNames[1]}' — missed cases`,
      "11": `${count} rows (${pct}%) were correctly predicted as '${classNames[1]}'`
    }
    return descriptions[`${row}${col}`]
  }

  function cellColour(row, col, value) {
    const intensity = value / maxVal
    const isCorrect = row === col
    const r = isCorrect ? 6   : 153
    const g = isCorrect ? 95  : 27
    const b = isCorrect ? 70  : 27
    return `rgba(${r},${g},${b},${0.08 + intensity * 0.75})`
  }

  return (
    <div>
      <div className="overflow-x-auto">
        {/* Column headers — predicted */}
        <div className="mb-2">
          <p className="text-xs text-gray-400 text-center mb-1">Predicted →</p>
          <div className="grid" style={{
            gridTemplateColumns: `80px repeat(${classNames.length}, 1fr)`
          }}>
            <div /> {/* spacer */}
            {classNames.map(name => (
              <div key={name}
                   className="text-center text-xs font-semibold text-gray-600
                              py-2 px-1 truncate">
                {name}
              </div>
            ))}
          </div>
        </div>

        {/* Matrix rows */}
        {matrix.map((row, ri) => (
          <div key={ri} className="grid mb-1" style={{
            gridTemplateColumns: `80px repeat(${classNames.length}, 1fr)`
          }}>
            {/* Row label — actual */}
            <div className="flex items-center justify-end pr-3 text-xs
                            font-semibold text-gray-600">
              {ri === 0 && (
                <span className="text-gray-400 text-xs mr-2"
                      style={{ writingMode: "vertical-lr",
                               transform: "rotate(180deg)" }}>
                  Actual ↑
                </span>
              )}
              {classNames[ri]}
            </div>

            {/* Cells */}
            {row.map((count, ci) => (
              <button
                key={ci}
                onMouseEnter={() => setHovered({ ri, ci, count })}
                onMouseLeave={() => setHovered(null)}
                className="relative aspect-square rounded-lg m-0.5 flex
                           flex-col items-center justify-center transition-all
                           duration-150 focus:outline-none focus:ring-2
                           focus:ring-blue-400"
                style={{ backgroundColor: cellColour(ri, ci, count) }}
              >
                <span className="text-base font-bold text-gray-800">
                  {count.toLocaleString()}
                </span>
                {classNames.length === 2 && (
                  <span className="text-xs text-gray-500 mt-0.5">
                    {cellLabel(ri, ci)}
                  </span>
                )}
              </button>
            ))}
          </div>
        ))}
      </div>

      {/* Hover description */}
      <div className={`mt-4 p-3 rounded-xl text-sm transition-all duration-200
                       ${hovered
                         ? "bg-gray-50 border border-gray-200 text-gray-700"
                         : "opacity-0"}`}>
        {hovered
          ? cellDescription(hovered.ri, hovered.ci, hovered.count)
          : "Hover a cell to understand what it means"}
      </div>
    </div>
  )
}
```

---

## 5. Tuning Trial History (Tuning)

Line chart of Optuna trial scores over time — shows the search converging.

```jsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid,
         Tooltip, ReferenceLine, ResponsiveContainer } from "recharts"

function TuningTrialChart({ trials, metricName, baselineScore }) {
  // trials: [{ trial: 1, score: 0.821 }, { trial: 2, score: 0.834 }, ...]

  // Compute running best
  const data = trials.reduce((acc, t, i) => {
    const prev    = acc[i - 1]?.runningBest ?? 0
    const runBest = Math.max(prev, t.score)
    return [...acc, { ...t, runningBest: runBest }]
  }, [])

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={tooltipStyle}>
        <p className="font-semibold text-gray-700">Trial {label}</p>
        <p className="text-gray-600">
          Score: <span className="font-mono">{payload[0]?.value?.toFixed(4)}</span>
        </p>
        <p className="text-blue-600">
          Best so far: <span className="font-mono">
            {payload[1]?.value?.toFixed(4)}
          </span>
        </p>
      </div>
    )
  }

  return (
    <div>
      <p className="text-xs text-gray-400 mb-3">
        Each dot is one combination of settings tried. The blue line shows
        the best score found so far. Hover for details.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ ...CHART_DEFAULTS.margin, left: 8 }}>
          <CartesianGrid stroke={CHART_COLOURS.grid} strokeDasharray="3 3" />
          <XAxis
            dataKey="trial"
            label={{ value: "Trial number", position: "insideBottom",
                     offset: -4, fontSize: 11, fill: "#9CA3AF" }}
            {...axisProps}
          />
          <YAxis
            domain={["auto", "auto"]}
            tickFormatter={v => v.toFixed(3)}
            {...axisProps}
            width={52}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Baseline reference line */}
          <ReferenceLine
            y={baselineScore}
            stroke={CHART_COLOURS.muted}
            strokeDasharray="4 4"
            label={{ value: "Baseline", position: "insideTopRight",
                     fontSize: 11, fill: "#9CA3AF" }}
          />

          {/* Individual trial scores */}
          <Line
            type="monotone"
            dataKey="score"
            stroke={CHART_COLOURS.muted}
            dot={{ r: 2, fill: CHART_COLOURS.muted }}
            strokeWidth={1}
            isAnimationActive={false}
          />

          {/* Running best */}
          <Line
            type="stepAfter"
            dataKey="runningBest"
            stroke={CHART_COLOURS.primary}
            dot={false}
            strokeWidth={CHART_DEFAULTS.strokeWidth}
            animationDuration={CHART_DEFAULTS.animationDuration}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
```

---

## 6. Feature Importance Bar (Explainability)

Interactive SHAP importance — hover for exact value and plain English meaning.

```jsx
function FeatureImportanceChart({ features }) {
  // features: [{ feature: "tenure", importance: 0.234 }, ...]
  // Already sorted descending

  const top = features.slice(0, 15)
  const maxVal = top[0]?.importance ?? 1

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    const relativeInfluence = ((d.importance / maxVal) * 100).toFixed(0)
    return (
      <div style={tooltipStyle}>
        <p className="font-semibold text-gray-800">{d.feature}</p>
        <p className="text-gray-600 text-xs mt-1">
          Average influence: <span className="font-mono">{d.importance.toFixed(4)}</span>
        </p>
        <p className="text-blue-600 text-xs">
          {relativeInfluence}% of the most influential feature
        </p>
      </div>
    )
  }

  return (
    <div>
      <p className="text-xs text-gray-400 mb-3">
        Longer bar = this feature has more influence on the model's predictions.
        Hover a bar for the exact value.
      </p>
      <ResponsiveContainer width="100%" height={Math.max(200, top.length * 32)}>
        <BarChart
          data={[...top].reverse()}
          layout="vertical"
          margin={{ top: 4, right: 48, bottom: 4, left: 8 }}
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
            width={120}
            {...axisProps}
            tick={{ ...axisProps.tick, fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "#F3F4F6" }} />
          <Bar
            dataKey="importance"
            fill={CHART_COLOURS.primary}
            fillOpacity={0.85}
            radius={[0, 4, 4, 0]}
            label={{
              position: "right",
              formatter: v => v.toFixed(3),
              fontSize: 11,
              fill: "#9CA3AF"
            }}
            animationDuration={CHART_DEFAULTS.animationDuration}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
```

---

## 7. Performance Trend Chart (Monitoring)

Time-series line chart showing the primary metric across monitoring reports.
This is the most important monitoring visualisation — static PNG cannot do this.

```jsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
         ReferenceLine, ReferenceArea, ResponsiveContainer } from "recharts"

function PerformanceTrendChart({ reports, metricName, baselineScore,
                                  warningThreshold, criticalThreshold }) {
  // reports: [{ reportNumber: 1, date: "2025-03-01", score: 0.889 }, ...]

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    const score = payload[0].value
    const change = score - baselineScore
    return (
      <div style={tooltipStyle}>
        <p className="font-semibold text-gray-700">Report {label}</p>
        <p className="text-gray-500 text-xs">{payload[0].payload.date}</p>
        <p className="text-gray-800 mt-1">
          {metricName}: <span className="font-mono font-semibold">
            {score.toFixed(4)}
          </span>
        </p>
        <p className={`text-xs mt-0.5 ${change >= 0 ? "text-green-600" : "text-red-600"}`}>
          {change >= 0 ? "+" : ""}{(change * 100).toFixed(1)}% vs baseline
        </p>
      </div>
    )
  }

  return (
    <div>
      <p className="text-xs text-gray-400 mb-3">
        Each point is one monitoring check. The dashed line is your model's
        score at deployment. Hover for details.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={reports}
                   margin={{ ...CHART_DEFAULTS.margin, left: 8 }}>
          <CartesianGrid stroke={CHART_COLOURS.grid} strokeDasharray="3 3" />
          <XAxis dataKey="reportNumber"
                 label={{ value: "Report number", position: "insideBottom",
                          offset: -4, fontSize: 11, fill: "#9CA3AF" }}
                 {...axisProps} />
          <YAxis domain={["auto", "auto"]}
                 tickFormatter={v => v.toFixed(3)}
                 {...axisProps} width={52} />
          <Tooltip content={<CustomTooltip />} />

          {/* Baseline */}
          <ReferenceLine
            y={baselineScore}
            stroke={CHART_COLOURS.primary}
            strokeDasharray="5 3"
            label={{ value: "Deployment score", position: "insideTopLeft",
                     fontSize: 11, fill: CHART_COLOURS.primary }}
          />

          {/* Warning zone */}
          {warningThreshold && (
            <ReferenceArea
              y1={criticalThreshold ?? 0}
              y2={warningThreshold}
              fill={CHART_COLOURS.warning}
              fillOpacity={0.06}
            />
          )}

          <Line
            type="monotone"
            dataKey="score"
            stroke={CHART_COLOURS.primary}
            strokeWidth={CHART_DEFAULTS.strokeWidth}
            dot={{ r: CHART_DEFAULTS.dotRadius,
                   fill: CHART_COLOURS.primary,
                   strokeWidth: 2,
                   stroke: "#fff" }}
            activeDot={{ r: 6 }}
            animationDuration={CHART_DEFAULTS.animationDuration}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
```

---

## 8. Drift Summary Donut (Monitoring)

```jsx
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts"

function DriftSummaryDonut({ total, drifted, highSeverity, mediumSeverity }) {
  const noDrift = total - drifted
  const lowSeverity = drifted - highSeverity - mediumSeverity

  const data = [
    { name: "No drift",       value: noDrift,      colour: CHART_COLOURS.success  },
    { name: "Low drift",      value: lowSeverity,   colour: "#D1FAE5"              },
    { name: "Medium drift",   value: mediumSeverity, colour: CHART_COLOURS.accent  },
    { name: "High drift",     value: highSeverity,  colour: CHART_COLOURS.error    }
  ].filter(d => d.value > 0)

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
      <div style={tooltipStyle}>
        <p className="font-semibold" style={{ color: d.colour }}>{d.name}</p>
        <p className="text-gray-600">{d.value} feature{d.value !== 1 ? "s" : ""}</p>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-6">
      <div className="flex-shrink-0">
        <ResponsiveContainer width={140} height={140}>
          <PieChart>
            <Pie
              data={data} cx="50%" cy="50%"
              innerRadius={42} outerRadius={62}
              paddingAngle={2} dataKey="value"
              animationDuration={CHART_DEFAULTS.animationDuration}
            >
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.colour} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="space-y-2">
        {data.map(d => (
          <div key={d.name} className="flex items-center gap-2 text-sm">
            <div className="w-3 h-3 rounded-sm flex-shrink-0"
                 style={{ backgroundColor: d.colour }} />
            <span className="text-gray-700">{d.name}</span>
            <span className="text-gray-400 ml-auto pl-4">
              {d.value} / {total}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
```

---

## Static PNG Display Wrapper

For all Python-generated charts (SHAP beeswarm, ROC curve, residual plots,
correlation heatmap). Consistent treatment — loads from backend, shows loading
state, renders with descriptive alt text.

```jsx
function StaticChart({ sessionId, chartPath, altText, caption }) {
  const [src,     setSrc]     = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(false)

  useEffect(() => {
    if (!chartPath) return
    const url = `http://localhost:8001/sessions/${sessionId}/charts?path=${encodeURIComponent(chartPath)}`
    fetch(url)
      .then(r => r.blob())
      .then(blob => {
        setSrc(URL.createObjectURL(blob))
        setLoading(false)
      })
      .catch(() => { setError(true); setLoading(false) })
  }, [sessionId, chartPath])

  if (loading) return (
    <div className="w-full aspect-video bg-gray-50 rounded-xl
                    flex items-center justify-center animate-pulse">
      <p className="text-sm text-gray-400">Loading chart…</p>
    </div>
  )

  if (error) return (
    <div className="w-full aspect-video bg-red-50 rounded-xl
                    flex items-center justify-center border border-red-100">
      <p className="text-sm text-red-400">Chart could not be loaded.</p>
    </div>
  )

  return (
    <figure className="w-full">
      <img
        src={src}
        alt={altText}
        className="w-full rounded-xl border border-gray-100"
      />
      {caption && (
        <figcaption className="text-xs text-gray-400 text-center mt-2 leading-relaxed">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
```

---

## Reference Files

- `references/chart-data-contracts.md` — exact JSON shapes the backend must return for each interactive chart
