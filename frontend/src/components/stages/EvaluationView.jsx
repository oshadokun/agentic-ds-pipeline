import { useEffect, useState } from "react"
import { usePipeline } from "../../contexts/PipelineContext"
import { useSession } from "../../contexts/SessionContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import StaticChart from "../charts/StaticChart"
import InteractiveConfusionMatrix from "../charts/InteractiveConfusionMatrix"

const VERDICT_CONFIG = {
  strong: { label: "Strong performance",    colour: "text-green-700", bg: "bg-green-50",  border: "border-green-200" },
  good:   { label: "Good performance",      colour: "text-blue-700",  bg: "bg-blue-50",   border: "border-blue-200" },
  fair:   { label: "Fair performance",      colour: "text-amber-700", bg: "bg-amber-50",  border: "border-amber-200" },
  poor:   { label: "Needs improvement",     colour: "text-red-700",   bg: "bg-red-50",    border: "border-red-200" }
}

export default function EvaluationView() {
  const { sessionId } = useSession()
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus
  } = usePipeline()

  const [hasLoaded, setHasLoaded] = useState(false)

  const stageStatus = getStageStatus("evaluation")
  const isComplete  = stageStatus === "complete"
  const result      = stageResult

  useEffect(() => {
    async function load() {
      if (isComplete) {
        await loadStageResult("evaluation")
      }
      setHasLoaded(true)
    }
    load()
  }, []) // eslint-disable-line

  if (stageRunning) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName="Evaluating model performance…"
          message="Running predictions on the held-out test data."
        />
      </div>
    )
  }

  const verdict = result?.verdict
  const vc      = VERDICT_CONFIG[verdict] ?? VERDICT_CONFIG.fair

  // Static charts from agent (e.g. ROC, PR curve, residuals)
  const staticCharts = (result?.charts ?? []).filter(c => !c.includes("confusion"))

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Evaluation failed" message={stageError} />
      )}

      {!isComplete && hasLoaded && (
        <div className="text-center py-8">
          <p className="text-gray-500 text-sm mb-4">
            We'll test the model on data it has never seen before, to get an honest measure of how well it works.
          </p>
          <button
            onClick={() => runStage("evaluation")}
            className="px-6 py-3 bg-[#1B3A5C] text-white rounded-xl font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Evaluate My Model
          </button>
        </div>
      )}

      {isComplete && result && (
        <div className="space-y-5">
          {/* Verdict — big and clear */}
          {verdict && (
            <div className={`rounded-2xl border p-6 ${vc.bg} ${vc.border}`}>
              <p className={`text-2xl font-serif font-bold ${vc.colour} mb-2`}>
                {vc.label}
              </p>
              <p className="text-gray-700 text-sm">
                {result.metric_name}:{" "}
                <span className="font-mono font-bold text-gray-900 text-base">
                  {result.test_score?.toFixed(4) ?? result.val_score?.toFixed(4) ?? "—"}
                </span>
                {" "}on the held-out test set
              </p>
              {result.plain_english_verdict && (
                <p className="text-sm text-gray-600 mt-2">{result.plain_english_verdict}</p>
              )}
            </div>
          )}

          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {/* Metric details */}
          {result.metrics && (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {Object.entries(result.metrics).map(([key, val]) => (
                <div key={key} className="bg-gray-50 rounded-xl p-3 text-center">
                  <p className="text-base font-semibold text-gray-800 font-mono">
                    {typeof val === "number" ? val.toFixed(4) : String(val)}
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">{key.replace(/_/g, " ")}</p>
                </div>
              ))}
            </div>
          )}

          {/* Confusion matrix */}
          {result.confusion_matrix && result.class_names && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-4">Prediction breakdown</h3>
              <InteractiveConfusionMatrix
                matrix={result.confusion_matrix}
                classNames={result.class_names}
              />
            </div>
          )}

          {/* Static charts (ROC, residuals, etc.) */}
          {staticCharts.map((chartPath, i) => (
            <div key={i} className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <StaticChart
                sessionId={sessionId}
                chartPath={chartPath}
                caption={
                  chartPath.includes("roc") ? "ROC curve — measures how well the model separates outcomes. Closer to the top-left corner is better." :
                  chartPath.includes("precision") ? "Precision-recall curve — shows the trade-off between catching all cases and avoiding false alarms." :
                  chartPath.includes("residual") ? "Residuals — difference between predicted and actual values. Points should be scattered randomly around zero." :
                  "Performance chart"
                }
              />
            </div>
          ))}
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete}
        continueLabel="Continue to Fine-Tune"
      />
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Measure Performance</h2>
      <p className="text-gray-500 text-sm">
        We'll test the model on data it has never seen before. This gives us an honest
        picture of how well it would perform on new data in the real world.
      </p>
    </div>
  )
}
