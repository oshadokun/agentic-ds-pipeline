import { useEffect, useState } from "react"
import { usePipeline } from "../../contexts/PipelineContext"
import { useSession } from "../../contexts/SessionContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import DecisionCard from "../shared/DecisionCard"
import FeatureImportanceChart from "../charts/FeatureImportanceChart"

export default function FeatureEngineeringView() {
  const { sessionId } = useSession()
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus,
    decisions, setDecisions
  } = usePipeline()

  const [hasLoaded, setHasLoaded] = useState(false)
  const [confirmed, setConfirmed] = useState({})
  const [applying, setApplying]   = useState(false)

  const stageStatus    = getStageStatus("feature_engineering")
  const isComplete     = stageStatus === "complete"
  const result         = stageResult
  const required       = result?.decisions_required ?? []
  const needsDecisions = required.length > 0 && result?.status === "decisions_required"
  const allConfirmed   = required.every(d => confirmed[d.id])

  useEffect(() => {
    async function load() {
      if (isComplete) {
        await loadStageResult("feature_engineering")
      } else {
        await runStage("feature_engineering")
      }
      setHasLoaded(true)
    }
    load()
  }, []) // eslint-disable-line

  function handleConfirm(decisionId, value) {
    setDecisions({ [decisionId]: value })
    setConfirmed(c => ({ ...c, [decisionId]: true }))
  }

  async function applyDecisions() {
    setApplying(true)
    await runStage("feature_engineering", decisions)
    setApplying(false)
  }

  if (stageRunning || applying) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName={applying ? "Transforming your features…" : "Analysing columns for encoding…"}
          message="This may take a moment."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Feature engineering failed" message={stageError} />
      )}

      {/* Encoding decisions */}
      {needsDecisions && hasLoaded && (
        <div className="space-y-4">
          <ExplanationPanel
            message={result.plain_english_summary ??
              `Your data contains text or category columns. The model can't use text directly — we need to convert them to numbers. Here is what we recommend for each:`}
          />

          {required.map(d => (
            <DecisionCard
              key={d.id}
              decision={d}
              onConfirm={handleConfirm}
            />
          ))}

          {allConfirmed && (
            <button
              onClick={applyDecisions}
              className="w-full py-3 rounded-xl bg-[#1B3A5C] text-white font-medium
                         hover:bg-[#162f4d] transition-colors text-sm"
            >
              Apply feature transformations
            </button>
          )}
        </div>
      )}

      {/* Auto run if no decisions */}
      {!needsDecisions && !isComplete && hasLoaded && !stageRunning && (
        <div className="text-center py-8">
          <p className="text-gray-500 text-sm mb-4">
            No encoding decisions are needed. We'll apply the recommended transformations automatically.
          </p>
          <button
            onClick={() => runStage("feature_engineering")}
            className="px-6 py-3 bg-[#1B3A5C] text-white rounded-xl font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Prepare My Features
          </button>
        </div>
      )}

      {/* Complete */}
      {isComplete && result && (
        <div className="space-y-5">
          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {/* Feature importance chart */}
          {result.feature_importance?.length > 0 && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-1">Most informative features</h3>
              <p className="text-xs text-gray-400 mb-4">
                Longer bar = greater influence on predicting your outcome.
              </p>
              <FeatureImportanceChart
                features={[...result.feature_importance].sort((a, b) => b.importance - a.importance)}
              />
            </div>
          )}

          {/* Actions taken */}
          {result.actions_taken?.length > 0 && (
            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">Transformations applied</p>
              <ul className="space-y-1.5">
                {result.actions_taken.map((action, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                    <span className="text-green-500 mt-0.5 flex-shrink-0">✓</span>
                    <span>{typeof action === "string" ? action : action.plain_english}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Feature counts */}
          {result.config_updates && (
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-50 rounded-xl p-3 text-center">
                <p className="text-lg font-semibold text-gray-800">{result.config_updates.features_selected ?? "—"}</p>
                <p className="text-xs text-gray-400 mt-0.5">Features kept</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3 text-center">
                <p className="text-lg font-semibold text-gray-800">{result.config_updates.features_dropped ?? "—"}</p>
                <p className="text-xs text-gray-400 mt-0.5">Features removed</p>
              </div>
            </div>
          )}
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete}
        continueLabel="Continue to Scale Data"
      />
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Prepare Features</h2>
      <p className="text-gray-500 text-sm">
        Models can only learn from numbers. We'll convert text columns, expand date columns,
        and select the most useful features for predicting your outcome.
      </p>
    </div>
  )
}
