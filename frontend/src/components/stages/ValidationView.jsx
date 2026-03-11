import { useEffect, useState } from "react"
import { CheckCircle, AlertTriangle, XCircle, Info, Calendar, ChevronDown, ChevronUp } from "lucide-react"
import { usePipeline } from "../../contexts/PipelineContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"

const SEVERITY_ICON = {
  hard_stop: <XCircle className="w-5 h-5 text-red-500 flex-shrink-0" />,
  warning:   <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0" />,
  advisory:  <Info className="w-5 h-5 text-blue-400 flex-shrink-0" />,
  pass:      <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
}

const SEVERITY_STYLES = {
  hard_stop: "border-red-200 bg-red-50",
  warning:   "border-amber-200 bg-amber-50",
  advisory:  "border-blue-100 bg-blue-50",
  pass:      "border-green-100 bg-white"
}

export default function ValidationView() {
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus
  } = usePipeline()

  const [hasLoaded, setHasLoaded]       = useState(false)
  const [summaryOpen, setSummaryOpen]   = useState(false)
  const [summaryTab, setSummaryTab]     = useState("columns") // "columns" | "stats"

  const stageStatus = getStageStatus("validation")
  const isComplete  = stageStatus === "complete"
  const result      = stageResult

  useEffect(() => {
    async function load() {
      if (isComplete) await loadStageResult("validation")
      setHasLoaded(true)
    }
    load()
  }, []) // eslint-disable-line

  const hasStop         = result?.hard_stop === true || result?.status === "hard_stop"
  const report          = result?.validation_report ?? {}
  const hardStops       = report.hard_stops  ?? []
  const warnings        = report.warnings    ?? []
  // Suppress date_column advisories — they are already surfaced as purple Date cards
  const advisories      = (report.advisories ?? []).filter(
    a => a.check !== "date_column"
  )
  const totalChecks     = report.total_checks_run ?? 0
  const failedCount     = hardStops.length + warnings.length + advisories.length
  const passedCount     = Math.max(0, totalChecks - failedCount)
  const timeSeriesCols  = result?.time_series_columns ?? []

  if (stageRunning) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName="Running data quality checks…"
          message="Checking for missing values, data types, structural issues, and more."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Check failed" message={stageError} />
      )}

      {/* Run button */}
      {!isComplete && hasLoaded && (
        <div className="text-center py-8">
          <p className="text-gray-500 text-sm mb-4">
            We'll run 6 checks on your data to make sure it's ready for modelling.
          </p>
          <button
            onClick={() => runStage("validation")}
            className="px-6 py-3 bg-[#1B3A5C] text-white rounded-xl font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Run Quality Checks
          </button>
        </div>
      )}

      {/* Results */}
      {isComplete && result && (
        <div className="space-y-4">
          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {hasStop && hardStops.length > 0 && (
            <AlertBanner
              type="error"
              title={`${hardStops.length} critical issue${hardStops.length > 1 ? "s" : ""} found`}
              message="These must be fixed before we can continue. Please correct your data file and re-upload it."
            />
          )}

          <div className="space-y-2">
            {[...hardStops, ...warnings, ...advisories].map((check, i) => {
              const sev = check.severity
              return (
                <div
                  key={i}
                  className={`flex items-start gap-3 p-4 rounded-xl border ${SEVERITY_STYLES[sev]}`}
                >
                  {SEVERITY_ICON[sev]}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-800">{check.message}</p>
                    {check.action && (
                      <p className="text-xs text-gray-500 mt-0.5">{check.action}</p>
                    )}
                  </div>
                  <span className={`text-xs font-medium px-2 py-0.5 rounded-full flex-shrink-0 ${
                    sev === "hard_stop" ? "bg-red-100 text-red-700"
                    : sev === "warning"  ? "bg-amber-100 text-amber-700"
                    : "bg-blue-100 text-blue-700"
                  }`}>
                    {sev === "hard_stop" ? "Critical" : sev.charAt(0).toUpperCase() + sev.slice(1)}
                  </span>
                </div>
              )
            })}
          </div>

          {/* Date / time columns found */}
          {timeSeriesCols.length > 0 && (
            <div className="space-y-2">
              {timeSeriesCols.map(col => (
                <div
                  key={col}
                  className="flex items-start gap-3 p-4 rounded-xl border border-purple-100 bg-purple-50"
                >
                  <Calendar className="w-5 h-5 text-purple-500 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-800">
                      Date column detected: <strong>{col}</strong>
                    </p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      We'll extract useful time features from this column automatically — month, year, day of week, and quarter.
                    </p>
                  </div>
                  <span className="text-xs font-medium px-2 py-0.5 rounded-full flex-shrink-0 bg-purple-100 text-purple-700">
                    Date
                  </span>
                </div>
              ))}
            </div>
          )}

          <div className="flex gap-4 text-xs pl-1">
            {passedCount > 0 && (
              <span className="text-green-600">{passedCount} check{passedCount !== 1 ? "s" : ""} passed</span>
            )}
            {warnings.length > 0 && (
              <span className="text-amber-600">{warnings.length} warning{warnings.length > 1 ? "s" : ""}</span>
            )}
            {hardStops.length > 0 && (
              <span className="text-red-600">{hardStops.length} critical</span>
            )}
          </div>
        </div>
      )}

      {isComplete && result?.data_summary && (
        <DataSummaryPanel
          summary={result.data_summary}
          open={summaryOpen}
          onToggle={() => setSummaryOpen(o => !o)}
          tab={summaryTab}
          onTabChange={setSummaryTab}
        />
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete || hasStop}
        continueLabel="Continue to Explore Data"
      />
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Check Data Quality</h2>
      <p className="text-gray-500 text-sm">
        Before we do anything else, we'll check your data for common problems —
        missing values, wrong types, and structural issues — so the model has reliable information to learn from.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Dtype display helper
// ---------------------------------------------------------------------------

function friendlyDtype(dtype) {
  if (!dtype) return "unknown"
  if (dtype.startsWith("int"))      return "integer"
  if (dtype.startsWith("float"))    return "decimal"
  if (dtype.startsWith("datetime")) return "date"
  if (dtype === "object")           return "text"
  if (dtype === "bool")             return "boolean"
  return dtype
}

// ---------------------------------------------------------------------------
// Data Summary Panel
// ---------------------------------------------------------------------------

function DataSummaryPanel({ summary, open, onToggle, tab, onTabChange }) {
  const { shape, columns = [], numeric_stats = [] } = summary

  return (
    <div className="border border-gray-200 rounded-2xl overflow-hidden bg-white shadow-sm">
      {/* Toggle header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-5 py-4
                   hover:bg-gray-50 transition-colors text-left"
      >
        <div>
          <span className="text-sm font-semibold text-gray-800">View data summary</span>
          <span className="text-xs text-gray-400 ml-2">
            {shape.rows.toLocaleString()} rows · {shape.columns} columns
          </span>
        </div>
        {open
          ? <ChevronUp className="w-4 h-4 text-gray-400 flex-shrink-0" />
          : <ChevronDown className="w-4 h-4 text-gray-400 flex-shrink-0" />
        }
      </button>

      {open && (
        <div className="border-t border-gray-100">
          {/* Tabs */}
          <div className="flex border-b border-gray-100">
            {["columns", "stats"].map(t => (
              <button
                key={t}
                onClick={() => onTabChange(t)}
                className={`px-5 py-2.5 text-xs font-medium transition-colors ${
                  tab === t
                    ? "border-b-2 border-[#1B3A5C] text-[#1B3A5C]"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {t === "columns" ? "Columns" : "Numeric stats"}
              </button>
            ))}
          </div>

          {/* Columns tab */}
          {tab === "columns" && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-gray-50 text-gray-500 text-left">
                    <th className="px-4 py-2.5 font-medium">Column</th>
                    <th className="px-4 py-2.5 font-medium">Type</th>
                    <th className="px-4 py-2.5 font-medium text-right">Non-null</th>
                    <th className="px-4 py-2.5 font-medium text-right">% missing</th>
                  </tr>
                </thead>
                <tbody>
                  {columns.map((col, i) => (
                    <tr key={col.name}
                        className={`border-t border-gray-50 ${i % 2 === 0 ? "" : "bg-gray-50/40"}`}>
                      <td className="px-4 py-2 font-mono text-gray-800 max-w-[180px] truncate">
                        {col.name}
                      </td>
                      <td className="px-4 py-2 text-gray-500">{friendlyDtype(col.dtype)}</td>
                      <td className="px-4 py-2 text-right text-gray-600">
                        {col.non_null_count.toLocaleString()}
                      </td>
                      <td className={`px-4 py-2 text-right font-medium ${
                        col.null_pct > 0.3 ? "text-red-600"
                        : col.null_pct > 0 ? "text-amber-600"
                        : "text-green-600"
                      }`}>
                        {col.null_pct > 0 ? `${(col.null_pct * 100).toFixed(1)}%` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Stats tab */}
          {tab === "stats" && (
            numeric_stats.length === 0
              ? <p className="px-5 py-4 text-sm text-gray-400">No numeric columns found.</p>
              : (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="bg-gray-50 text-gray-500 text-left">
                        <th className="px-4 py-2.5 font-medium">Column</th>
                        <th className="px-4 py-2.5 font-medium text-right">Min</th>
                        <th className="px-4 py-2.5 font-medium text-right">Max</th>
                        <th className="px-4 py-2.5 font-medium text-right">Mean</th>
                        <th className="px-4 py-2.5 font-medium text-right">Median</th>
                        <th className="px-4 py-2.5 font-medium text-right">Std dev</th>
                      </tr>
                    </thead>
                    <tbody>
                      {numeric_stats.map((s, i) => (
                        <tr key={s.name}
                            className={`border-t border-gray-50 ${i % 2 === 0 ? "" : "bg-gray-50/40"}`}>
                          <td className="px-4 py-2 font-mono text-gray-800 max-w-[180px] truncate">
                            {s.name}
                          </td>
                          {["min", "max", "mean", "median", "std"].map(k => (
                            <td key={k} className="px-4 py-2 text-right text-gray-600 font-mono">
                              {s[k] != null ? s[k].toLocaleString(undefined, { maximumFractionDigits: 4 }) : "—"}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )
          )}
        </div>
      )}
    </div>
  )
}
