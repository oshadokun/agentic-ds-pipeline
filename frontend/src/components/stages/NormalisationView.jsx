import { useEffect, useState } from "react"
import { usePipeline } from "../../contexts/PipelineContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import DecisionCard from "../shared/DecisionCard"

export default function NormalisationView() {
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus,
    decisions, setDecisions
  } = usePipeline()

  const [hasLoaded, setHasLoaded]   = useState(false)
  const [confirmed, setConfirmed]   = useState(false)
  const [applying, setApplying]     = useState(false)

  const stageStatus    = getStageStatus("normalisation")
  const isComplete     = stageStatus === "complete"
  const result         = stageResult
  const required       = result?.decisions_required ?? []
  const needsDecisions = required.length > 0 && result?.status === "decisions_required"

  useEffect(() => {
    async function load() {
      let res
      if (isComplete) {
        res = await loadStageResult("normalisation")
      } else {
        res = await runStage("normalisation")
      }
      console.log("[NormalisationView] stage result:", res)
      console.log("[NormalisationView] decisions_required:", res?.decisions_required)
      console.log("[NormalisationView] recommendation:", res?.decisions_required?.[0]?.recommendation)
      setHasLoaded(true)
    }
    load()
  }, []) // eslint-disable-line

  function handleConfirm(decisionId, value) {
    setDecisions({ [decisionId]: value })
    setConfirmed(true)
  }

  async function applyDecisions() {
    setApplying(true)
    await runStage("normalisation", decisions)
    setApplying(false)
  }

  if (stageRunning || applying) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName={applying ? "Scaling your data…" : "Checking data ranges…"}
          message="Fitting the scaler to your feature columns."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Scaling failed" message={stageError} />
      )}

      {/* Scaler decision */}
      {needsDecisions && hasLoaded && (
        <div className="space-y-4">
          <ExplanationPanel
            message={result.plain_english_summary ??
              "Different features in your data are on very different scales — for example, age (18–80) and salary (20,000–200,000). Scaling brings them to a comparable range so the model doesn't unfairly weight the larger numbers."}
          />

          {required.map(d => (
            <DecisionCard
              key={`${d.id}-${d.recommendation}`}
              decision={d}
              onConfirm={handleConfirm}
            />
          ))}

          {confirmed && (
            <button
              onClick={applyDecisions}
              className="w-full py-3 rounded-xl bg-[#1B3A5C] text-white font-medium
                         hover:bg-[#162f4d] transition-colors text-sm"
            >
              Apply scaling
            </button>
          )}
        </div>
      )}

      {/* Auto run */}
      {!needsDecisions && !isComplete && hasLoaded && !stageRunning && (
        <div className="text-center py-8">
          <p className="text-gray-500 text-sm mb-4">
            We'll scale your numeric columns to a consistent range.
          </p>
          <button
            onClick={() => runStage("normalisation")}
            className="px-6 py-3 bg-[#1B3A5C] text-white rounded-xl font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Scale My Data
          </button>
        </div>
      )}

      {/* Complete */}
      {isComplete && result && (
        <div className="space-y-4">
          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm space-y-3">
            {result.scaler_type && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">Scaler used</span>
                <span className="text-sm font-semibold text-gray-800 capitalize">
                  {result.scaler_type.replace("_", " ")}
                </span>
              </div>
            )}
            {result.features_scaled !== undefined && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">Columns scaled</span>
                <span className="text-sm font-semibold text-gray-800">{result.features_scaled}</span>
              </div>
            )}
            <p className="text-xs text-gray-400 pt-2 border-t border-gray-50">
              The scaler has been fitted on your full dataset and saved. It will be applied consistently to new data at prediction time.
            </p>
          </div>
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete}
        continueLabel="Continue to Divide Data"
      />
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Scale Your Data</h2>
      <p className="text-gray-500 text-sm">
        We'll bring all your numeric columns to a similar scale. This prevents larger numbers
        from dominating the model's learning process.
      </p>
    </div>
  )
}
