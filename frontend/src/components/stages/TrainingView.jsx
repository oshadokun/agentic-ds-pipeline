import { useEffect, useState } from "react"
import { usePipeline } from "../../contexts/PipelineContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import DecisionCard from "../shared/DecisionCard"

const VERDICT_STYLES = {
  strong: { bg: "bg-green-50",  border: "border-green-200",  text: "text-green-800",  label: "Strong result" },
  good:   { bg: "bg-blue-50",   border: "border-blue-200",   text: "text-blue-800",   label: "Good result" },
  fair:   { bg: "bg-amber-50",  border: "border-amber-200",  text: "text-amber-800",  label: "Fair result" },
  poor:   { bg: "bg-red-50",    border: "border-red-200",    text: "text-red-800",    label: "Needs improvement" }
}

export default function TrainingView() {
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus,
    decisions, setDecisions, clearStageError
  } = usePipeline()

  const [hasLoaded, setHasLoaded] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const [applying, setApplying]   = useState(false)

  const stageStatus    = getStageStatus("training")
  const isComplete     = stageStatus === "complete"
  const result         = stageResult
  const required       = result?.decisions_required ?? []
  const needsDecisions = required.length > 0 && result?.status === "decisions_required"

  useEffect(() => {
    clearStageError()   // clear any stale error from a previous attempt
    if (isComplete) {
      loadStageResult("training").then(() => setHasLoaded(true))
    } else {
      setHasLoaded(true)  // show the button — don't auto-trigger the backend
    }
  }, []) // eslint-disable-line

  function handleConfirm(decisionId, value) {
    setDecisions({ [decisionId]: value })
    setConfirmed(true)
  }

  async function applyDecisions() {
    setApplying(true)
    await runStage("training", decisions)
    setApplying(false)
  }

  if (stageRunning || applying) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName={applying ? "Training your model…" : "Selecting the best approach…"}
          message="The model is learning patterns from your training data. This may take a minute."
        />
      </div>
    )
  }

  const verdict = result?.verdict ?? result?.baseline_verdict
  const verdictStyle = VERDICT_STYLES[verdict] ?? VERDICT_STYLES.fair

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Training failed" message={stageError} />
      )}

      {/* Model selection */}
      {needsDecisions && hasLoaded && (
        <div className="space-y-4">
          <ExplanationPanel
            message={result.plain_english_summary ??
              "We'll start with a baseline model to see how well the data predicts the outcome. First, choose which type of model to train:"}
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
              Train the model
            </button>
          )}
        </div>
      )}

      {/* Auto start */}
      {!needsDecisions && !isComplete && hasLoaded && !stageRunning && (
        <div className="text-center py-8">
          <p className="text-gray-500 text-sm mb-4">
            Ready to train a baseline model using the recommended approach.
          </p>
          <button
            onClick={() => runStage("training")}
            className="px-6 py-3 bg-[#1B3A5C] text-white rounded-xl font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Train My Model
          </button>
        </div>
      )}

      {/* Results */}
      {isComplete && result && (
        <div className="space-y-4">
          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {/* Verdict banner */}
          {verdict && (
            <div className={`rounded-2xl border p-5 ${verdictStyle.bg} ${verdictStyle.border}`}>
              <p className={`text-lg font-bold ${verdictStyle.text} mb-1`}>
                {verdictStyle.label}
              </p>
              <p className="text-sm text-gray-600">
                {result.metric_name}: <span className="font-mono font-semibold">{result.val_score?.toFixed(4)}</span>
                {" "}on validation data
              </p>
            </div>
          )}

          {/* Score cards */}
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "Training score",   value: result.train_score?.toFixed(4) },
              { label: "Validation score", value: result.val_score?.toFixed(4) },
              { label: "Model type",       value: result.model_type?.replace("_", " ") },
              { label: "Metric",           value: result.metric_name }
            ].map(({ label, value }) => (
              <div key={label} className="bg-gray-50 rounded-xl p-3">
                <p className="text-xs text-gray-400 mb-0.5">{label}</p>
                <p className="font-semibold text-gray-800 capitalize">{value ?? "—"}</p>
              </div>
            ))}
          </div>

          {/* Overfitting warning — only meaningful for accuracy/R² metrics, not MAE/RMSE */}
          {result.train_score && result.val_score &&
           result.metric_name !== "MAE" && result.metric_name !== "RMSE" &&
           (result.train_score - result.val_score) > 0.1 && (
            <AlertBanner
              type="warning"
              title="Possible overfitting"
              message="The model scores noticeably higher on training data than validation data. This is normal — we'll address it during fine-tuning."
            />
          )}
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete}
        continueLabel="Continue to Measure Performance"
      />
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Train the Model</h2>
      <p className="text-gray-500 text-sm">
        We'll train a model on your data and see how well it learns your outcome.
        This gives us a baseline to improve from.
      </p>
    </div>
  )
}
