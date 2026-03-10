import { useState, useEffect } from "react"
import { CheckCircle } from "lucide-react"

/**
 * DecisionCard
 * General-purpose decision component. Supports single-choice and numeric input.
 *
 * Props:
 *   decision — the decision object from the backend:
 *     { id, question, recommendation, recommendation_reason, alternatives: [{ id, label, tradeoff }] }
 *   onConfirm(decisionId, chosenValue) — called when user confirms
 *   disabled — disables interaction (e.g. already confirmed)
 */
export default function DecisionCard({ decision, onConfirm, disabled = false }) {
  const { id, question, recommendation, recommendation_reason, alternatives = [] } = decision
  const [selected, setSelected] = useState(recommendation ?? alternatives[0]?.id ?? null)
  const [confirmed, setConfirmed] = useState(false)

  // Keep selected in sync if recommendation arrives after initial mount
  useEffect(() => {
    if (recommendation !== undefined && recommendation !== null) {
      setSelected(recommendation)
    }
  }, [recommendation])

  function handleConfirm() {
    if (selected === null || selected === undefined) return
    setConfirmed(true)
    onConfirm(id, selected)
  }

  // Confirmed state — shows a compact summary row
  if (confirmed) {
    const chosen = alternatives.find(a => a.id === selected)
    const chosenLabel = chosen?.label ?? chosen?.name ?? String(selected)
    return (
      <div className="flex items-center gap-3 p-3 bg-green-50 border border-green-200
                      rounded-xl mb-4 animate-fade-in">
        <CheckCircle size={16} className="text-green-600 flex-shrink-0" />
        <div className="min-w-0">
          <span className="text-sm text-green-800 font-medium">{question}</span>
          <span className="text-sm text-green-700 ml-2">→ {chosenLabel}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 mb-5 shadow-sm animate-slide-up">
      {/* Question */}
      <p className="font-medium text-gray-900 mb-1 text-base">{question}</p>

      {/* Recommendation reason */}
      {recommendation_reason && (
        <p className="text-sm text-gray-500 mb-4 leading-relaxed">
          {recommendation_reason}
        </p>
      )}

      {/* Alternatives */}
      <div className="space-y-2.5 mb-5">
        {alternatives.map(opt => {
          const isSelected = selected === opt.id
          return (
            <button
              key={opt.id}
              onClick={() => !disabled && setSelected(opt.id)}
              disabled={disabled}
              className={`
                w-full text-left p-4 rounded-xl border-2 transition-all duration-150
                ${isSelected
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50"}
                ${disabled ? "cursor-default" : "cursor-pointer"}
              `}
            >
              <div className="flex items-start gap-3">
                {/* Radio dot */}
                <div
                  className={`
                    mt-0.5 w-5 h-5 rounded-full border-2 flex-shrink-0
                    flex items-center justify-center
                    ${isSelected ? "border-blue-500" : "border-gray-300"}
                  `}
                >
                  {isSelected && (
                    <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-0.5">
                    <span className="font-medium text-gray-900 text-sm">{opt.label ?? opt.name}</span>
                    {opt.id === recommendation && (
                      <span className="text-xs px-2 py-0.5 bg-amber-100
                                       text-amber-700 rounded-full font-medium">
                        Recommended
                      </span>
                    )}
                  </div>
                  {opt.tradeoff && (
                    <p className="text-xs text-gray-400 mt-0.5 leading-relaxed italic">
                      Trade-off: {opt.tradeoff}
                    </p>
                  )}
                </div>
              </div>
            </button>
          )
        })}
      </div>

      {/* Confirm button */}
      <button
        onClick={handleConfirm}
        disabled={disabled || selected === null || selected === undefined}
        className="px-6 py-2.5 bg-[#1B3A5C] text-white text-sm font-medium
                   rounded-xl hover:bg-[#2E6099] disabled:opacity-40
                   disabled:cursor-not-allowed transition-colors"
      >
        Confirm Choice
      </button>
    </div>
  )
}
