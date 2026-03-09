/**
 * GoalCaptureView
 * The very first screen — asks the user to describe their goal in plain English,
 * then creates a session and navigates into the pipeline.
 * Phase 5 will expand this with the full interaction flow.
 */

import { useState } from "react"
import { Brain, ArrowRight } from "lucide-react"
import { useSession } from "../../contexts/SessionContext"
import { usePipeline } from "../../contexts/PipelineContext"
import AlertBanner from "../shared/AlertBanner"

const EXAMPLES = [
  "I want to predict which customers are likely to cancel their subscription.",
  "I'd like to forecast monthly sales revenue.",
  "I want to classify product reviews as positive, neutral, or negative."
]

export default function GoalCaptureView() {
  const { createSession, error, clearError } = useSession()
  const { goToStage } = usePipeline()

  const [goal, setGoal]       = useState("")
  const [loading, setLoading] = useState(false)

  const MIN_LEN = 20
  const tooShort = goal.trim().length < MIN_LEN

  async function handleSubmit() {
    if (tooShort || loading) return
    setLoading(true)
    const session = await createSession(goal.trim())
    setLoading(false)
    if (session) {
      goToStage("ingestion")
    }
  }

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-[#1B3A5C] flex items-center justify-center">
          <Brain size={20} className="text-white" />
        </div>
        <h1 className="text-2xl font-serif text-gray-900">What do you want to predict?</h1>
      </div>

      <p className="text-gray-600 leading-relaxed mb-6">
        Describe your goal in plain English. You do not need to use technical terms.
        The more specific you are, the better we can guide you.
      </p>

      {error && (
        <AlertBanner type="error" message={error} onClose={clearError} />
      )}

      {/* Text input */}
      <div className="mb-6">
        <textarea
          value={goal}
          onChange={e => { setGoal(e.target.value); clearError?.() }}
          placeholder="e.g. I want to predict which customers are likely to churn…"
          rows={4}
          className={`
            w-full px-4 py-3 rounded-xl border text-base text-gray-800
            resize-none focus:outline-none focus:ring-2 focus:ring-blue-400
            transition-shadow leading-relaxed bg-white
            ${error ? "border-red-300" : "border-gray-200"}
          `}
        />
        <div className="flex items-center justify-between mt-2">
          <span className={`text-xs ${tooShort ? "text-gray-300" : "text-gray-500"}`}>
            {goal.trim().length} characters
            {tooShort && ` — please write at least ${MIN_LEN}`}
          </span>
          <button
            onClick={handleSubmit}
            disabled={tooShort || loading}
            className="flex items-center gap-2 px-5 py-2.5 bg-[#1B3A5C] text-white
                       rounded-xl text-sm font-medium hover:bg-[#2E6099]
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Creating…" : "Begin Pipeline"}
            {!loading && <ArrowRight size={16} />}
          </button>
        </div>
      </div>

      {/* Examples */}
      <div>
        <p className="text-xs text-gray-400 mb-2 font-medium uppercase tracking-wide">
          Examples
        </p>
        <div className="space-y-2">
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              onClick={() => setGoal(ex)}
              className="w-full text-left px-4 py-3 rounded-xl border border-gray-200
                         text-sm text-gray-600 hover:border-blue-300 hover:bg-blue-50
                         transition-colors"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
