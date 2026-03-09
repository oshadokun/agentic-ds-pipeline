import { useState } from "react"
import { Brain, HelpCircle } from "lucide-react"

/**
 * ExplanationPanel
 * Renders the Orchestrator's plain-English narration in a conversation bubble.
 * Always shows a "Tell me more" affordance.
 *
 * Props:
 *   message      — the main plain-English text
 *   onFollowUp   — async fn(question) — called if user asks a follow-up
 *   animate      — whether to animate in (default true)
 */
export default function ExplanationPanel({ message, onFollowUp, animate = true }) {
  const [open, setOpen]     = useState(false)
  const [question, setQ]    = useState("")
  const [loading, setLoad]  = useState(false)

  async function handleAsk() {
    if (!question.trim() || !onFollowUp) return
    setLoad(true)
    await onFollowUp(question.trim())
    setQ("")
    setOpen(false)
    setLoad(false)
  }

  return (
    <div className={`flex gap-4 mb-6 ${animate ? "animate-slide-up" : ""}`}>
      {/* Avatar */}
      <div
        className="flex-shrink-0 w-9 h-9 rounded-xl bg-[#1B3A5C]
                   flex items-center justify-center"
        aria-hidden="true"
      >
        <Brain size={18} className="text-white" />
      </div>

      {/* Bubble */}
      <div className="flex-1 bg-white rounded-2xl rounded-tl-sm
                      border border-gray-100 shadow-sm px-5 py-4">
        <p className="text-xs font-semibold text-[#1B3A5C] mb-2 uppercase tracking-wide">
          Pipeline Guide
        </p>

        <div className="text-gray-700 text-base leading-relaxed whitespace-pre-line">
          {message}
        </div>

        {/* Follow-up affordance */}
        {onFollowUp && (
          <button
            onClick={() => setOpen(o => !o)}
            className="mt-3 text-sm text-blue-600 hover:text-blue-800
                       flex items-center gap-1.5 transition-colors"
          >
            <HelpCircle size={14} />
            {open ? "Never mind" : "I'd like more detail"}
          </button>
        )}

        {/* Follow-up input */}
        {open && (
          <div className="mt-3 flex gap-2 animate-slide-up">
            <input
              type="text"
              value={question}
              onChange={e => setQ(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleAsk()}
              placeholder="What would you like to know more about?"
              autoFocus
              className="flex-1 px-4 py-2 text-sm rounded-xl border border-gray-200
                         focus:outline-none focus:ring-2 focus:ring-blue-300 bg-gray-50"
            />
            <button
              onClick={handleAsk}
              disabled={!question.trim() || loading}
              className="px-4 py-2 bg-[#1B3A5C] text-white text-sm rounded-xl
                         disabled:opacity-40 hover:bg-[#2E6099] transition-colors"
            >
              {loading ? "…" : "Ask"}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
