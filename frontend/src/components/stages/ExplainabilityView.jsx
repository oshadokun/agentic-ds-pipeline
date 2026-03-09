import { useEffect, useState } from "react"
import { AlertTriangle } from "lucide-react"
import { usePipeline } from "../../contexts/PipelineContext"
import { useSession } from "../../contexts/SessionContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import StaticChart from "../charts/StaticChart"
import FeatureImportanceChart from "../charts/FeatureImportanceChart"

export default function ExplainabilityView() {
  const { sessionId } = useSession()
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus
  } = usePipeline()

  const [hasLoaded, setHasLoaded]       = useState(false)
  const [waterfallIndex, setWaterfallIndex] = useState(0)

  const stageStatus = getStageStatus("explainability")
  const isComplete  = stageStatus === "complete"
  const result      = stageResult

  useEffect(() => {
    async function load() {
      if (isComplete) {
        await loadStageResult("explainability")
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
          stageName="Calculating feature contributions…"
          message="Using SHAP values to understand how each feature influences predictions."
        />
      </div>
    )
  }

  // Waterfall charts from agent (individual prediction explanations)
  const waterfallCharts = (result?.charts ?? []).filter(c => c.includes("waterfall"))
  const summaryChart    = (result?.charts ?? []).find(c => c.includes("beeswarm") || c.includes("summary"))
  const biasFlags       = result?.bias_check?.flagged_columns ?? []

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Explainability failed" message={stageError} />
      )}

      {!isComplete && hasLoaded && (
        <div className="text-center py-8">
          <p className="text-gray-500 text-sm mb-4">
            We'll use SHAP values to show you exactly which features drive the model's predictions,
            and by how much.
          </p>
          <button
            onClick={() => runStage("explainability")}
            className="px-6 py-3 bg-[#1B3A5C] text-white rounded-xl font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Explain My Model
          </button>
        </div>
      )}

      {isComplete && result && (
        <div className="space-y-5">
          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {/* Bias warning */}
          {biasFlags.length > 0 && (
            <div className="border border-amber-200 bg-amber-50 rounded-2xl p-5">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-semibold text-amber-800 mb-1">Potential bias detected</p>
                  <p className="text-amber-700 text-sm">
                    {result.bias_check?.plain_english ??
                      `The following columns are among the most influential features and may contain sensitive attributes: ${biasFlags.join(", ")}. Review whether this is appropriate for your use case.`}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Feature importance chart */}
          {result.feature_importance?.length > 0 && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-1">
                Overall feature importance
              </h3>
              <p className="text-xs text-gray-400 mb-4">
                How much each feature contributes to the model's predictions on average.
              </p>
              <FeatureImportanceChart features={result.feature_importance} />
            </div>
          )}

          {/* SHAP beeswarm summary */}
          {summaryChart && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-4">SHAP summary — how features push predictions</h3>
              <StaticChart
                sessionId={sessionId}
                chartPath={summaryChart}
                caption="Each dot is one data point. Red = high feature value, blue = low. Dots on the right push predictions higher; dots on the left push them lower."
              />
            </div>
          )}

          {/* Individual prediction explanations */}
          {waterfallCharts.length > 0 && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-800">Individual prediction examples</h3>
                <div className="flex gap-1">
                  {waterfallCharts.map((_, i) => (
                    <button
                      key={i}
                      onClick={() => setWaterfallIndex(i)}
                      className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                        i === waterfallIndex
                          ? "bg-[#1B3A5C] text-white"
                          : "border border-gray-200 text-gray-500 hover:bg-gray-50"
                      }`}
                    >
                      Example {i + 1}
                    </button>
                  ))}
                </div>
              </div>
              <StaticChart
                sessionId={sessionId}
                chartPath={waterfallCharts[waterfallIndex]}
                caption="Each bar shows how much a single feature pushed this particular prediction up (right) or down (left) from the average."
              />
            </div>
          )}
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete}
        continueLabel="Continue to Deploy Model"
      />
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Understand the Model</h2>
      <p className="text-gray-500 text-sm">
        We'll use SHAP values to show you exactly which features the model relies on,
        and how each one influences a prediction.
      </p>
    </div>
  )
}
