import { useEffect, useState } from "react"
import { usePipeline } from "../../contexts/PipelineContext"
import { useSession } from "../../contexts/SessionContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import StaticChart from "../charts/StaticChart"
import InteractiveConfusionMatrix from "../charts/InteractiveConfusionMatrix"
import TimeSeriesChart from "../charts/TimeSeriesChart"

const VERDICT_CONFIG = {
  strong: { label: "Strong performance",    colour: "text-green-700", bg: "bg-green-50",  border: "border-green-200" },
  good:   { label: "Good performance",      colour: "text-blue-700",  bg: "bg-blue-50",   border: "border-blue-200" },
  fair:   { label: "Fair performance",      colour: "text-amber-700", bg: "bg-amber-50",  border: "border-amber-200" },
  poor:   { label: "Needs improvement",     colour: "text-red-700",   bg: "bg-red-50",    border: "border-red-200" }
}

// Keys to always hide from the metric grid
const HIDDEN_METRIC_KEYS = new Set(["confusion_matrix", "predictions"])

// Plain-English explanations for every metric
const METRIC_META = {
  accuracy:    { label: "Accuracy",     explain: v => `We correctly predicted the outcome for ${(v*100).toFixed(1)}% of rows.` },
  precision:   { label: "Precision",    explain: v => `When the model predicted a positive outcome, it was right ${(v*100).toFixed(1)}% of the time.` },
  recall:      { label: "Recall",       explain: v => `We correctly identified ${(v*100).toFixed(1)}% of all actual positive cases.` },
  f1:          { label: "F1 Score",     explain: v => `Balance between precision and recall — ${v.toFixed(3)} out of 1.0.` },
  roc_auc:     { label: "AUC-ROC",      explain: v => `The model separates classes with a score of ${v.toFixed(3)} (1.0 = perfect, 0.5 = random guess).` },
  pr_auc:      { label: "AUC-PR",       explain: v => `Precision-Recall AUC: ${v.toFixed(3)} — especially important for imbalanced data where AUC-ROC can be misleadingly high.` },
  mcc:         { label: "MCC",          explain: v => `Matthews Correlation Coefficient: ${v.toFixed(3)} (−1 to +1, higher is better). The best single metric for imbalanced classification.` },
  specificity: { label: "Specificity",  explain: v => `We correctly identified ${(v*100).toFixed(1)}% of all actual negative cases.` },
  log_loss:    { label: "Log Loss",     explain: v => `Measures confidence of predictions: ${v.toFixed(4)} — lower is better.` },
  mae:         { label: "MAE",          explain: v => `On average, predictions are off by ${v.toFixed(4)} (in the same units as the target).` },
  rmse:        { label: "RMSE",         explain: v => `Root Mean Squared Error: ${v.toFixed(4)} — penalises large errors more heavily than MAE.` },
  r2:          { label: "R²",           explain: v => `The model explains ${(v*100).toFixed(1)}% of the variation in the target.` },
  mape:        { label: "MAPE",         explain: v => `On average, predictions are off by ${v.toFixed(1)}% relative to the actual value.` },
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
  const isTs    = result?.is_time_series ?? false

  // Static charts — for TS models, skip the static time_series chart since the
  // interactive chart below already shows actual vs predicted
  const staticCharts = (result?.charts ?? []).filter(c => {
    if (c.includes("confusion")) return false
    if (isTs && c.includes("time_series")) return false
    return true
  })

  // Metric grid: skip internal/non-numeric keys, raw arrays, and hidden keys
  const metricEntries = Object.entries(result?.metrics ?? {}).filter(([key, val]) => {
    if (HIDDEN_METRIC_KEYS.has(key)) return false
    if (Array.isArray(val)) return false
    if (typeof val !== "number") return false
    return true
  })

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
                {result.primary_metric_name?.toUpperCase() ?? "Score"}:{" "}
                <span className="font-mono font-bold text-gray-900 text-base">
                  {result.primary_metric_value != null
                    ? result.primary_metric_value.toFixed(4)
                    : "—"}
                </span>
                {" "}on the held-out {result.split_evaluated ?? "validation"} set
              </p>
              {result.verdict_message && (
                <p className="text-sm text-gray-600 mt-2">{result.verdict_message}</p>
              )}
            </div>
          )}

          {result.high_r2_warning && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
              <p className="text-sm font-semibold text-amber-800 mb-1">Almost-perfect R² — worth investigating</p>
              <p className="text-sm text-amber-700 leading-relaxed">
                Your model's R² looks almost perfect. This sometimes happens when the model
                has learned patterns in your training data that won't hold on truly new data —
                a problem called <strong>data leakage</strong>. This is worth investigating
                before deploying.
              </p>
            </div>
          )}

          {result.high_auc_warning && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
              <p className="text-sm font-semibold text-amber-800 mb-1">Almost-perfect detection — worth investigating</p>
              <p className="text-sm text-amber-700 leading-relaxed">
                Your model's detection looks almost perfect. On heavily imbalanced data this can
                sometimes mean the model has overfit. Review the{" "}
                <strong>Precision-Recall score</strong> and{" "}
                <strong>Confusion Matrix</strong> for a more honest picture.
              </p>
            </div>
          )}

          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {/* Metric details with plain-English explanations */}
          {metricEntries.length > 0 && (
            <div className="space-y-2">
              {metricEntries.map(([key, val]) => {
                const meta = METRIC_META[key]
                return (
                  <div key={key} className="bg-gray-50 rounded-xl p-4 flex items-start gap-4">
                    <div className="text-right min-w-[80px]">
                      <p className="text-xl font-bold text-gray-900 font-mono leading-none">
                        {val.toFixed(4)}
                      </p>
                      <p className="text-xs text-gray-400 mt-0.5">
                        {meta?.label ?? key.replace(/_/g, " ")}
                      </p>
                    </div>
                    <p className="text-sm text-gray-600 leading-relaxed pt-0.5 flex-1">
                      {meta ? meta.explain(val) : `${key.replace(/_/g, " ")}: ${val.toFixed(4)}`}
                    </p>
                  </div>
                )
              })}
            </div>
          )}

          {/* Confusion matrix */}
          {result.metrics?.confusion_matrix && result.class_names && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-4">Prediction breakdown</h3>
              <InteractiveConfusionMatrix
                matrix={result.metrics.confusion_matrix}
                classNames={result.class_names}
              />
            </div>
          )}

          {/* Time series actual vs predicted (interactive) */}
          {result.time_series_data?.length > 0 && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-1">Actual vs Predicted over time</h3>
              <p className="text-xs text-gray-400 mb-4">
                Blue line = what actually happened. Amber dashed = what the model predicted.
                Closer together = better accuracy.
              </p>
              <TimeSeriesChart data={result.time_series_data} />
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
