import { useEffect, useState } from "react"
import { Activity, RefreshCw, AlertTriangle, CheckCircle, XCircle, Download, Code } from "lucide-react"
import { usePipeline } from "../../contexts/PipelineContext"
import { useSession } from "../../contexts/SessionContext"
import { api } from "../../api"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import DriftSummaryDonut from "../charts/DriftSummaryDonut"
import PerformanceTrendChart from "../charts/PerformanceTrendChart"


const HEALTH_CONFIG = {
  healthy:          { icon: <CheckCircle className="w-7 h-7 text-green-500" />,  label: "All Good",         bg: "bg-green-50",  border: "border-green-200",  text: "text-green-800" },
  warning:          { icon: <AlertTriangle className="w-7 h-7 text-amber-500" />, label: "Warning",          bg: "bg-amber-50",  border: "border-amber-200",  text: "text-amber-800" },
  critical:         { icon: <XCircle className="w-7 h-7 text-red-500" />,        label: "Attention Needed", bg: "bg-red-50",    border: "border-red-200",    text: "text-red-800" },
  attention_needed: { icon: <XCircle className="w-7 h-7 text-red-500" />,        label: "Attention Needed", bg: "bg-red-50",    border: "border-red-200",    text: "text-red-800" }
}

const DRIFT_STATUS_STYLES = {
  no_drift: "text-green-600",
  low:      "text-blue-600",
  medium:   "text-amber-600",
  high:     "text-red-600"
}

export default function MonitoringView() {
  const { sessionId } = useSession()
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    getStageStatus
  } = usePipeline()

  const [hasLoaded, setHasLoaded] = useState(false)
  const [running, setRunning]     = useState(false)

  const stageStatus = getStageStatus("monitoring")
  const isComplete  = stageStatus === "complete"
  const result      = stageResult

  useEffect(() => {
    async function load() {
      if (isComplete) await loadStageResult("monitoring")
      setHasLoaded(true)
    }
    load()
  }, []) // eslint-disable-line

  async function runCheck() {
    setRunning(true)
    await runStage("monitoring")
    setRunning(false)
  }

  if (stageRunning || running) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName="Running monitoring check…"
          message="Comparing current data and predictions against the deployment baseline."
        />
      </div>
    )
  }

  const driftedCount = result?.drifted_features_count ?? result?.drifted_features ?? 0
  const totalFeatures = result?.total_features ?? 0
  const highSeverity  = result?.high_severity_drift ?? result?.high_severity ?? 0
  const healthStatus  = result?.health_status ?? (
    highSeverity > 0 ? "critical" :
    driftedCount > 0 ? "warning" : "healthy"
  )
  const hc           = HEALTH_CONFIG[healthStatus] ?? HEALTH_CONFIG.healthy
  const driftDetails = result?.drift_results ?? result?.drift_details ?? []
  const reports      = result?.report_history ?? []

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Monitoring check failed" message={stageError} />
      )}

      {/* First-run — establish baseline */}
      {!isComplete && hasLoaded && (
        <div className="space-y-4">
          <ExplanationPanel
            message="Monitoring watches your model over time to catch when the data it's receiving starts to look different from what it was trained on — and when its performance starts to drop. We'll establish a baseline now."
          />
          <button
            onClick={runCheck}
            className="w-full py-3 rounded-xl bg-[#1B3A5C] text-white font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Establish Baseline &amp; Run First Check
          </button>
        </div>
      )}

      {/* Results */}
      {isComplete && result && (
        <div className="space-y-5">
          {/* Health status — big and clear */}
          <div className={`rounded-2xl border p-5 ${hc.bg} ${hc.border}`}>
            <div className="flex items-center gap-3">
              {hc.icon}
              <div>
                <p className={`text-xl font-serif font-bold ${hc.text}`}>{hc.label}</p>
                <p className="text-sm text-gray-600 mt-0.5">
                  {result.plain_english_summary ?? "Model monitoring check complete."}
                </p>
              </div>
            </div>
          </div>

          {/* Retraining recommendation */}
          {result.recommend_retraining && (
            <AlertBanner
              type="warning"
              title="Retraining recommended"
              message={result.recommendation ?? "Significant drift or performance decay detected. Consider retraining the model on more recent data."}
            />
          )}

          {/* Drift donut */}
          {totalFeatures > 0 && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-4">Data drift summary</h3>
              <DriftSummaryDonut
                total={totalFeatures}
                drifted={driftedCount}
                highSeverity={highSeverity}
                mediumSeverity={result?.medium_severity ?? 0}
              />
            </div>
          )}

          {/* Performance trend */}
          {reports.length > 0 && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-1">Performance over time</h3>
              <PerformanceTrendChart
                reports={reports}
                metricName={result.metric_name ?? "Score"}
                baselineScore={result.baseline_score}
                warningThreshold={result.warning_threshold}
                criticalThreshold={result.critical_threshold}
              />
            </div>
          )}

          {/* Feature-by-feature drift table */}
          {driftDetails.length > 0 && (
            <div className="border border-gray-100 rounded-2xl overflow-hidden bg-white shadow-sm">
              <div className="px-5 py-4 border-b border-gray-50">
                <h3 className="font-semibold text-gray-800">Feature drift details</h3>
              </div>
              <div className="divide-y divide-gray-50">
                {driftDetails.map((item, i) => (
                  <div key={i} className="px-5 py-3 flex items-start gap-4">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-800">{item.feature}</p>
                      {item.plain_english && (
                        <p className="text-xs text-gray-400 mt-0.5">{item.plain_english}</p>
                      )}
                    </div>
                    <span className={`text-xs font-semibold capitalize flex-shrink-0 ${
                      DRIFT_STATUS_STYLES[item.status] ?? "text-gray-500"
                    }`}>
                      {item.status?.replace("_", " ") ?? "unknown"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Run check again */}
          <button
            onClick={runCheck}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-gray-200
                       text-sm text-gray-600 hover:bg-gray-50 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Run check again
          </button>

          {result.report_number && (
            <p className="text-xs text-gray-400">
              Report #{result.report_number} &middot; {result.date ?? ""}
            </p>
          )}
        </div>
      )}

      {/* Pipeline complete — download report + view code */}
      {isComplete && (
        <PipelineCompletePanel sessionId={sessionId} />
      )}

      <StageNavigation
        onContinue={() => {}}
        continueDisabled={true}
        continueLabel="Pipeline complete"
      />
    </div>
  )
}

function PipelineCompletePanel({ sessionId }) {
  function triggerDownload(url) {
    const a    = document.createElement("a")
    a.href     = url
    a.download = ""
    a.click()
  }

  return (
    <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm space-y-3">
      <h3 className="font-semibold text-gray-800">Your pipeline is complete</h3>
      <p className="text-sm text-gray-500">
        Download a summary report of everything that was done, or export the analysis
        code to run and modify it yourself.
      </p>
      <div className="flex flex-wrap gap-3">
        <button
          onClick={() => triggerDownload(api.reportDownloadUrl(sessionId))}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl
                     bg-[#1B3A5C] text-white text-sm font-medium
                     hover:bg-[#162f4d] transition-colors"
        >
          <Download className="w-4 h-4" />
          Download Report
        </button>
        <button
          onClick={() => triggerDownload(api.scriptDownloadUrl(sessionId))}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl
                     border border-gray-200 text-sm text-gray-700 font-medium
                     hover:bg-gray-50 transition-colors"
        >
          <Code className="w-4 h-4" />
          Download .py Script
        </button>
        <button
          onClick={() => triggerDownload(api.notebookDownloadUrl(sessionId))}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl
                     border border-gray-200 text-sm text-gray-700 font-medium
                     hover:bg-gray-50 transition-colors"
        >
          <Code className="w-4 h-4" />
          Download .ipynb Notebook
        </button>
      </div>
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Monitor Performance</h2>
      <p className="text-gray-500 text-sm">
        Track your model over time. We'll alert you if the data it's seeing starts to drift
        away from what it was trained on, or if its performance is declining.
      </p>
    </div>
  )
}
