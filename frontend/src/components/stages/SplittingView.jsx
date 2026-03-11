import { useEffect, useState } from "react"
import { usePipeline } from "../../contexts/PipelineContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import DecisionCard from "../shared/DecisionCard"
import SplitRatioDiagram from "../charts/SplitRatioDiagram"

export default function SplittingView() {
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus,
    decisions, setDecisions
  } = usePipeline()

  const [hasLoaded, setHasLoaded]         = useState(false)
  const [confirmedIds, setConfirmedIds]   = useState({})
  const [applying, setApplying]           = useState(false)
  const [ratios, setRatios]               = useState(null)

  const stageStatus    = getStageStatus("splitting")
  const isComplete     = stageStatus === "complete"
  const result         = stageResult
  const required       = result?.decisions_required ?? []
  const needsDecisions = required.length > 0 && result?.status === "decisions_required"
  const allConfirmed   = required.every(d => confirmedIds[d.id])

  useEffect(() => {
    async function load() {
      if (isComplete) {
        await loadStageResult("splitting")
      } else {
        await runStage("splitting")
      }
      setHasLoaded(true)
    }
    load()
  }, []) // eslint-disable-line

  function handleConfirm(decisionId, value) {
    setDecisions({ [decisionId]: value })
    setConfirmedIds(c => ({ ...c, [decisionId]: true }))
  }

  function handleRatioChange(newRatios) {
    setRatios(newRatios)
    setDecisions({ split_ratios: newRatios })
  }

  async function applyDecisions() {
    setApplying(true)
    await runStage("splitting", decisions)
    setApplying(false)
  }

  // Derive recommended ratios from the backend response (percentages, e.g. {train:75,val:10,test:15})
  const defaultRatios = result?.recommended_ratios ?? { train: 70, val: 15, test: 15 }

  if (stageRunning || applying) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName={applying ? "Splitting your data…" : "Calculating recommended split…"}
          message="Dividing your data into training, validation, and test sets."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Split failed" message={stageError} />
      )}

      {needsDecisions && hasLoaded && (
        <div className="space-y-4">
          <ExplanationPanel
            message={result.plain_english_summary ??
              "We need to divide your data into three sets: one to train the model on, one to tune it, and one kept completely hidden until the final test."}
          />

          {/* Split ratio diagram */}
          <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
            <h3 className="font-semibold text-gray-800 mb-1">Adjust your split ratios</h3>
            <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200
                           rounded-xl px-3 py-2 mb-4">
              These are our recommended ratios based on your dataset size. You can adjust them freely below.
            </p>
            <SplitRatioDiagram
              defaultRatios={defaultRatios}
              onChange={handleRatioChange}
              totalRows={result?.total_rows}
            />
          </div>

          {/* Other decisions (strategy etc.) */}
          {required.filter(d => d.id !== "split_ratios").map(d => (
            <DecisionCard
              key={d.id}
              decision={d}
              onConfirm={handleConfirm}
            />
          ))}

          <button
            onClick={applyDecisions}
            className="w-full py-3 rounded-xl bg-[#1B3A5C] text-white font-medium
                       hover:bg-[#162f4d] transition-colors text-sm"
          >
            Split my data
          </button>
        </div>
      )}

      {/* Auto run */}
      {!needsDecisions && !isComplete && hasLoaded && !stageRunning && (
        <div className="text-center py-8">
          <p className="text-gray-500 text-sm mb-4">
            We'll divide your data using the recommended split.
          </p>
          <button
            onClick={() => runStage("splitting")}
            className="px-6 py-3 bg-[#1B3A5C] text-white rounded-xl font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Split My Data
          </button>
        </div>
      )}

      {/* Complete */}
      {isComplete && result && (
        <div className="space-y-4">
          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
            {(() => {
              const sizes = result.split_sizes ?? {}
              const total = (sizes.train ?? 0) + (sizes.val ?? 0) + (sizes.test ?? 0)
              return (
                <SplitRatioDiagram
                  defaultRatios={{
                    train: total ? Math.round((sizes.train ?? 0) / total * 100) : 70,
                    val:   total ? Math.round((sizes.val   ?? 0) / total * 100) : 15,
                    test:  total ? Math.round((sizes.test  ?? 0) / total * 100) : 15,
                  }}
                  totalRows={total || result.total_rows}
                  disabled
                />
              )
            })()}
          </div>

          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Train rows", value: result.split_sizes?.train },
              { label: "Val rows",   value: result.split_sizes?.val },
              { label: "Test rows",  value: result.split_sizes?.test }
            ].map(({ label, value }) => (
              <div key={label} className="bg-gray-50 rounded-xl p-3 text-center">
                <p className="text-lg font-semibold text-gray-800">
                  {value != null ? value.toLocaleString() : "—"}
                </p>
                <p className="text-xs text-gray-400 mt-0.5">{label}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete}
        continueLabel="Continue to Train Model"
      />
    </div>
  )
}

function parseRatios(str) {
  const parts = String(str).split("/").map(Number)
  if (parts.length === 3 && parts.every(n => !isNaN(n))) {
    return { train: parts[0], val: parts[1], test: parts[2] }
  }
  return { train: 70, val: 15, test: 15 }
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Divide Your Data</h2>
      <p className="text-gray-500 text-sm">
        We'll split your data into three parts: one to train the model, one to tune it,
        and one kept completely hidden for the final honest test.
      </p>
    </div>
  )
}
