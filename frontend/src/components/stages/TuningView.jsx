import { useEffect, useState } from "react"
import { usePipeline } from "../../contexts/PipelineContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import DecisionCard from "../shared/DecisionCard"
import TuningTrialChart from "../charts/TuningTrialChart"

export default function TuningView() {
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus,
    decisions, setDecisions
  } = usePipeline()

  const [hasLoaded, setHasLoaded] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const [applying, setApplying]   = useState(false)

  const stageStatus    = getStageStatus("tuning")
  const isComplete     = stageStatus === "complete"
  const result         = stageResult
  const required       = result?.decisions_required ?? []
  const needsDecisions = required.length > 0 && result?.status === "decisions_required"

  useEffect(() => {
    async function load() {
      if (isComplete) {
        await loadStageResult("tuning")
      } else {
        await runStage("tuning")
      }
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
    await runStage("tuning", decisions)
    setApplying(false)
  }

  const improvement = result?.best_score && result?.baseline_score
    ? ((result.best_score - result.baseline_score) * 100).toFixed(2)
    : null

  if (stageRunning || applying) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName="Testing different model settings…"
          message="Running through combinations of settings to find the best configuration. This may take a few minutes."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Tuning failed" message={stageError} />
      )}

      {/* n_trials decision */}
      {needsDecisions && hasLoaded && (
        <div className="space-y-4">
          <ExplanationPanel
            message={result.plain_english_summary ??
              "Fine-tuning tries different combinations of settings to find the ones that make the model perform best. More trials = more thorough search, but takes longer."}
          />

          {required.map(d => (
            <DecisionCard
              key={d.id}
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
              Start fine-tuning
            </button>
          )}
        </div>
      )}

      {/* Auto start */}
      {!needsDecisions && !isComplete && hasLoaded && !stageRunning && (
        <div className="text-center py-8">
          <p className="text-gray-500 text-sm mb-4">
            Ready to run automatic hyperparameter search.
          </p>
          <button
            onClick={() => runStage("tuning")}
            className="px-6 py-3 bg-[#1B3A5C] text-white rounded-xl font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Fine-Tune My Model
          </button>
        </div>
      )}

      {/* Results */}
      {isComplete && result && (
        <div className="space-y-5">
          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {/* Before / after */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <p className="text-xs text-gray-400 mb-1">Before tuning</p>
              <p className="text-lg font-semibold text-gray-600 font-mono">
                {result.baseline_score?.toFixed(4) ?? "—"}
              </p>
            </div>
            <div className="bg-blue-50 rounded-xl p-3 text-center border border-blue-100">
              <p className="text-xs text-blue-500 mb-1">After tuning</p>
              <p className="text-lg font-semibold text-[#1B3A5C] font-mono">
                {result.best_score?.toFixed(4) ?? "—"}
              </p>
            </div>
            <div className={`rounded-xl p-3 text-center ${
              parseFloat(improvement) > 0 ? "bg-green-50" : "bg-gray-50"
            }`}>
              <p className="text-xs text-gray-400 mb-1">Improvement</p>
              <p className={`text-lg font-semibold font-mono ${
                parseFloat(improvement) > 0 ? "text-green-700" : "text-gray-500"
              }`}>
                {improvement !== null
                  ? `${parseFloat(improvement) >= 0 ? "+" : ""}${improvement}%`
                  : "—"}
              </p>
            </div>
          </div>

          {/* Trial chart */}
          {result.trials?.length > 0 && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-1">Search progress</h3>
              <TuningTrialChart
                trials={result.trials}
                metricName={result.metric_name ?? "Score"}
                baselineScore={result.baseline_score}
              />
            </div>
          )}

          {/* Best params */}
          {result.best_params && Object.keys(result.best_params).length > 0 && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-3">Best settings found</h3>
              <div className="space-y-2">
                {Object.entries(result.best_params).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between text-sm">
                    <span className="text-gray-500">{k.replace(/_/g, " ")}</span>
                    <span className="font-mono font-medium text-gray-800">{String(v)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete}
        continueLabel="Continue to Understand Model"
      />
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Fine-Tune the Model</h2>
      <p className="text-gray-500 text-sm">
        We'll automatically search for the best model settings by trying many combinations
        and keeping the one that performs best.
      </p>
    </div>
  )
}
