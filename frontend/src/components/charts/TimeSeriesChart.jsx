import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from "recharts"

const COLOURS = { actual: "#1B3A5C", predicted: "#D97706" }

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-md px-3 py-2 text-xs">
      <p className="text-gray-400 mb-1">Step {label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }} className="font-medium">
          {p.name}: {Number(p.value).toFixed(4)}
        </p>
      ))}
    </div>
  )
}

/**
 * TimeSeriesChart — actual vs predicted line chart for time series evaluation.
 *
 * Props:
 *   data  — array of { index, actual, predicted }
 *   label — optional label for the y-axis (defaults to "Value")
 */
export default function TimeSeriesChart({ data = [], label = "Value" }) {
  if (!data.length) return null

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
        <XAxis
          dataKey="index"
          tick={{ fontSize: 11, fill: "#9CA3AF" }}
          tickLine={false}
          axisLine={{ stroke: "#E5E7EB" }}
          label={{ value: "Time step", position: "insideBottom", offset: -2, fontSize: 11, fill: "#9CA3AF" }}
        />
        <YAxis
          tick={{ fontSize: 11, fill: "#9CA3AF" }}
          tickLine={false}
          axisLine={false}
          width={60}
          label={{ value: label, angle: -90, position: "insideLeft", fontSize: 11, fill: "#9CA3AF" }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          iconType="line"
          wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
        />
        <Line
          type="monotone"
          dataKey="actual"
          name="Actual"
          stroke={COLOURS.actual}
          strokeWidth={1.8}
          dot={false}
          activeDot={{ r: 4 }}
        />
        <Line
          type="monotone"
          dataKey="predicted"
          name="Predicted"
          stroke={COLOURS.predicted}
          strokeWidth={1.8}
          strokeDasharray="5 3"
          dot={false}
          activeDot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
