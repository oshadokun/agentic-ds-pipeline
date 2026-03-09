import { useEffect, useState } from "react"
import { CheckCircle, Circle, Copy, Rocket, Code, ChevronDown, ChevronUp } from "lucide-react"
import { usePipeline } from "../../contexts/PipelineContext"
import { useSession } from "../../contexts/SessionContext"
import { api } from "../../api"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"

export default function DeploymentView() {
  const { session } = useSession()
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus
  } = usePipeline()

  const [hasLoaded, setHasLoaded]     = useState(false)
  const [copied, setCopied]           = useState(false)
  const [apiCode, setApiCode]         = useState(null)
  const [showCode, setShowCode]       = useState(false)
  const [codeCopied, setCodeCopied]   = useState(false)

  const stageStatus = getStageStatus("deployment")
  const isComplete  = stageStatus === "complete"
  const result      = stageResult

  // Pre-deployment checklist
  const trainStatus  = getStageStatus("training")
  const evalStatus   = getStageStatus("evaluation")
  const splitStatus  = getStageStatus("splitting")

  const checks = [
    { label: "Model trained",           done: trainStatus === "complete" },
    { label: "Model evaluated",         done: evalStatus === "complete" },
    { label: "Data scaler saved",       done: splitStatus === "complete" },
    { label: "All pipeline stages run", done: trainStatus === "complete" && evalStatus === "complete" }
  ]
  const readyToDeploy = checks.every(c => c.done)

  useEffect(() => {
    async function load() {
      if (isComplete) {
        await loadStageResult("deployment")
        try {
          const { code } = await api.getApiCode(sessionId)
          setApiCode(code)
        } catch { /* not yet generated */ }
      }
      setHasLoaded(true)
    }
    load()
  }, []) // eslint-disable-line

  function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const runCmd    = result?.run_command ?? "uvicorn app:app --port 8000"
  const curlExample = `curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"data": {${(result?.feature_columns ?? []).slice(0, 2).map(f => `"${f}": 0`).join(", ")}}}'`

  if (stageRunning) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName="Deploying your model…"
          message="Generating the API, building the container, and running a health check."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Deployment failed" message={stageError} />
      )}

      {/* Pre-deployment checklist */}
      {!isComplete && hasLoaded && (
        <div className="space-y-4">
          <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
            <h3 className="font-semibold text-gray-800 mb-4">Pre-deployment checklist</h3>
            <div className="space-y-3">
              {checks.map(({ label, done }) => (
                <div key={label} className="flex items-center gap-3">
                  {done
                    ? <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                    : <Circle className="w-5 h-5 text-gray-300 flex-shrink-0" />
                  }
                  <span className={`text-sm ${done ? "text-gray-800" : "text-gray-400"}`}>
                    {label}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {!readyToDeploy && (
            <AlertBanner
              type="warning"
              title="Not all stages complete"
              message="Complete all pipeline stages before deploying."
            />
          )}

          {readyToDeploy && (
            <>
              <ExplanationPanel
                message="Deployment creates a live API endpoint for your model. Anyone with the URL can send it new data and get instant predictions back."
              />
              <button
                onClick={async () => {
                  await runStage("deployment")
                  try {
                    const { code } = await api.getApiCode(sessionId)
                    setApiCode(code)
                  } catch { /* ignore */ }
                }}
                className="w-full py-3.5 rounded-xl bg-[#1B3A5C] text-white font-medium
                           hover:bg-[#162f4d] transition-colors flex items-center justify-center gap-2"
              >
                <Rocket className="w-4 h-4" />
                Deploy My Model
              </button>
            </>
          )}
        </div>
      )}

      {/* Success state */}
      {isComplete && result && (
        <div className="space-y-5">
          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {/* Packaged files card */}
          <div className="border border-green-200 bg-green-50 rounded-2xl p-5 space-y-3">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <span className="text-green-800 font-semibold text-sm">Model packaged and ready</span>
            </div>
            <p className="text-sm text-green-700">
              Your model, API code, Dockerfile, and requirements are saved in{" "}
              <code className="bg-green-100 px-1 rounded text-xs">{result.api_dir}</code>
            </p>
          </div>

          {/* Run command */}
          <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
            <h3 className="font-semibold text-gray-800 mb-2">Start the API</h3>
            <p className="text-xs text-gray-400 mb-3">
              Run this command in your terminal to start the prediction server:
            </p>
            <div className="flex items-center gap-2 bg-gray-900 rounded-xl px-4 py-3">
              <span className="text-gray-100 text-xs font-mono flex-1 overflow-x-auto">{runCmd}</span>
              <button
                onClick={() => copyToClipboard(runCmd)}
                className="flex-shrink-0 p-1 rounded hover:bg-gray-700 transition-colors"
                title="Copy command"
              >
                <Copy className="w-3.5 h-3.5 text-gray-400" />
              </button>
            </div>
            {copied && <p className="text-xs text-green-600 mt-1">Copied!</p>}
          </div>

          {/* Example curl */}
          <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
            <h3 className="font-semibold text-gray-800 mb-2">Example prediction request</h3>
            <p className="text-xs text-gray-400 mb-3">
              Once the API is running, send requests like this:
            </p>
            <pre className="bg-gray-900 text-gray-100 rounded-xl p-4 text-xs overflow-x-auto
                            leading-relaxed whitespace-pre-wrap">
              {curlExample}
            </pre>
          </div>

          {/* Test prediction result */}
          {result.test_prediction_result && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-3">Test prediction (verified)</h3>
              <pre className="bg-gray-50 rounded-xl p-3 text-xs font-mono text-gray-700 overflow-x-auto">
                {JSON.stringify(result.test_prediction_result, null, 2)}
              </pre>
            </div>
          )}

          {/* Generated API code */}
          {apiCode && (
            <div className="border border-gray-100 rounded-2xl bg-white shadow-sm overflow-hidden">
              <button
                onClick={() => setShowCode(c => !c)}
                className="w-full flex items-center justify-between px-5 py-4
                           hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Code className="w-4 h-4 text-gray-400" />
                  <span className="font-semibold text-gray-800 text-sm">View generated code (app.py)</span>
                </div>
                {showCode
                  ? <ChevronUp className="w-4 h-4 text-gray-400" />
                  : <ChevronDown className="w-4 h-4 text-gray-400" />
                }
              </button>

              {showCode && (
                <div className="border-t border-gray-100">
                  <div className="flex justify-end px-4 pt-3">
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(apiCode)
                        setCodeCopied(true)
                        setTimeout(() => setCodeCopied(false), 2000)
                      }}
                      className="flex items-center gap-1.5 text-xs text-gray-400
                                 hover:text-gray-600 px-2 py-1 rounded-lg
                                 hover:bg-gray-50 transition-colors"
                    >
                      <Copy className="w-3 h-3" />
                      {codeCopied ? "Copied!" : "Copy all"}
                    </button>
                  </div>
                  <pre className="bg-gray-900 text-gray-100 mx-4 mb-4 rounded-xl p-4
                                   text-xs overflow-x-auto leading-relaxed max-h-96
                                   overflow-y-auto whitespace-pre">
                    {apiCode}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete}
        continueLabel="Continue to Monitor Performance"
      />
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Deploy the Model</h2>
      <p className="text-gray-500 text-sm">
        We'll create a live API endpoint for your model so it can receive new data and
        return predictions in real time.
      </p>
    </div>
  )
}
