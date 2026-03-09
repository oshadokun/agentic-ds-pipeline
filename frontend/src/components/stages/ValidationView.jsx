import { useEffect, useState } from "react"
import { CheckCircle, AlertTriangle, XCircle, Info } from "lucide-react"
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

  const [hasLoaded, setHasLoaded] = useState(false)

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

  const checks     = result?.checks ?? []
  const hasStop    = result?.hard_stop === true || result?.status === "hard_stop"
  const hardStops  = checks.filter(c => !c.passed && c.severity === "hard_stop")
  const warnings   = checks.filter(c => !c.passed && c.severity === "warning")
  const advisories = checks.filter(c => !c.passed && c.severity === "advisory")
  const passed     = checks.filter(c => c.passed)

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

          {hasStop && (
            <AlertBanner
              type="error"
              title={`${hardStops.length} critical issue${hardStops.length > 1 ? "s" : ""} found`}
              message="These must be fixed before we can continue. Please correct your data file and re-upload it."
            />
          )}

          <div className="space-y-2">
            {[...hardStops, ...warnings, ...advisories, ...passed].map((check, i) => {
              const sev = !check.passed ? check.severity : "pass"
              return (
                <div
                  key={i}
                  className={`flex items-start gap-3 p-4 rounded-xl border ${SEVERITY_STYLES[sev]}`}
                >
                  {SEVERITY_ICON[sev]}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-800">{check.plain_english}</p>
                    {check.recommendation && !check.passed && (
                      <p className="text-xs text-gray-500 mt-0.5">{check.recommendation}</p>
                    )}
                  </div>
                  <span className={`text-xs font-medium px-2 py-0.5 rounded-full flex-shrink-0 ${
                    sev === "hard_stop" ? "bg-red-100 text-red-700"
                    : sev === "warning"  ? "bg-amber-100 text-amber-700"
                    : sev === "advisory" ? "bg-blue-100 text-blue-700"
                    : "bg-green-100 text-green-700"
                  }`}>
                    {sev === "pass" ? "Passed" : sev === "hard_stop" ? "Critical" : sev.charAt(0).toUpperCase() + sev.slice(1)}
                  </span>
                </div>
              )
            })}
          </div>

          <div className="flex gap-4 text-xs pl-1">
            <span className="text-green-600">{passed.length} passed</span>
            {warnings.length > 0 && (
              <span className="text-amber-600">{warnings.length} warning{warnings.length > 1 ? "s" : ""}</span>
            )}
            {hardStops.length > 0 && (
              <span className="text-red-600">{hardStops.length} critical</span>
            )}
          </div>
        </div>
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
