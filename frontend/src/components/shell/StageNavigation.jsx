import { ChevronLeft, ChevronRight, Loader2 } from "lucide-react"
import { usePipeline } from "../../contexts/PipelineContext"
import { STAGE_ORDER } from "../../contexts/stageConfig"

/**
 * StageNavigation
 * Back / Continue buttons shown at the bottom of each stage view.
 * Props:
 *   onContinue       — called when user clicks Continue (required)
 *   continueLabel    — override the button label (default: "Continue →")
 *   continueDisabled — disables the continue button
 *   showBack         — show the back button (default: true)
 *   hideNavigation   — hide the whole bar (e.g. during loading)
 */
export default function StageNavigation({
  onContinue,
  continueLabel    = "Continue →",
  continueDisabled = false,
  showBack         = true,
  hideNavigation   = false
}) {
  const { currentStage, stageRunning, goToPrevStage } = usePipeline()

  const isFirst = STAGE_ORDER.indexOf(currentStage) === 0

  if (hideNavigation) return null

  return (
    <div className="flex items-center justify-between pt-6 mt-6 border-t border-gray-100">
      {/* Back */}
      {showBack && !isFirst ? (
        <button
          onClick={goToPrevStage}
          disabled={stageRunning}
          className="flex items-center gap-2 px-4 py-2.5 text-sm text-gray-500
                     hover:text-gray-800 border border-gray-200 rounded-xl
                     hover:bg-gray-50 disabled:opacity-40 transition-colors"
        >
          <ChevronLeft size={16} />
          Back
        </button>
      ) : (
        <div /> /* spacer */
      )}

      {/* Continue */}
      <button
        onClick={onContinue}
        disabled={continueDisabled || stageRunning}
        className="flex items-center gap-2 px-6 py-2.5 bg-[#1B3A5C] text-white
                   text-sm font-medium rounded-xl hover:bg-[#2E6099]
                   disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {stageRunning && <Loader2 size={16} className="animate-spin" />}
        {continueLabel}
        {!stageRunning && <ChevronRight size={16} />}
      </button>
    </div>
  )
}
