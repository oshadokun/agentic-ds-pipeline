/**
 * chartTheme.js
 * Shared design tokens for all Recharts interactive charts.
 * Import from this file — never hard-code colours or styles in chart components.
 */

export const CHART_COLOURS = {
  primary:   "#1B3A5C",
  secondary: "#2E6099",
  accent:    "#D97706",
  success:   "#065F46",
  warning:   "#92400E",
  error:     "#991B1B",
  muted:     "#9CA3AF",
  grid:      "#F3F4F6",
  // Sequential palette for multi-series charts
  series: ["#1B3A5C", "#2E6099", "#D97706", "#065F46", "#7C3AED"]
}

export const CHART_DEFAULTS = {
  margin:            { top: 16, right: 16, bottom: 16, left: 16 },
  fontSize:          13,
  fontFamily:        "'DM Sans', system-ui, sans-serif",
  strokeWidth:       2,
  dotRadius:         4,
  animationDuration: 400
}

export const tooltipStyle = {
  backgroundColor: "#fff",
  border:          "1px solid #E5E7EB",
  borderRadius:    "10px",
  boxShadow:       "0 4px 16px rgba(0,0,0,0.08)",
  fontSize:        "13px",
  fontFamily:      "'DM Sans', system-ui, sans-serif",
  padding:         "8px 12px"
}

export const axisProps = {
  tick:     { fontSize: 12, fill: "#6B7280" },
  axisLine: { stroke: "#E5E7EB" },
  tickLine: { stroke: "#E5E7EB" }
}
