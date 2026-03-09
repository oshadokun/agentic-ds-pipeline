import {
  CheckCircle, Circle, ChevronRight,
  Upload, ShieldCheck, BarChart2, Sparkles, Wrench,
  Sliders, Scissors, Brain, Target, SlidersHorizontal,
  Lightbulb, Rocket, Activity, Trash2
} from "lucide-react"
import { usePipeline } from "../../contexts/PipelineContext"
import { STAGE_ORDER, STAGE_LABELS } from "../../contexts/stageConfig"
import { useSession } from "../../contexts/SessionContext"

// Map icon string names to components
const ICON_MAP = {
  Upload, ShieldCheck, BarChart2, Sparkles, Wrench,
  Sliders, Scissors, Brain, Target, SlidersHorizontal,
  Lightbulb, Rocket, Activity
}

function StageItem({ stage, status, label, iconName, isActive, onClick }) {
  const Icon = ICON_MAP[iconName] ?? Circle

  const styleMap = {
    complete:    "text-green-800 bg-green-50 border-green-200",
    in_progress: "text-blue-900 bg-blue-50 border-blue-300",
    failed:      "text-red-800 bg-red-50 border-red-200",
    pending:     "text-gray-400 bg-white border-gray-100"
  }

  const style = styleMap[status] ?? styleMap.pending

  return (
    <button
      onClick={status === "complete" || status === "in_progress" ? onClick : undefined}
      disabled={status === "pending"}
      aria-current={isActive ? "step" : undefined}
      className={`
        w-full flex items-center gap-3 px-3 py-2.5 rounded-xl border
        transition-all duration-150 text-left text-sm
        ${style}
        ${isActive ? "ring-2 ring-blue-400 ring-offset-1 font-semibold" : ""}
        ${status === "pending" ? "cursor-default opacity-60" : "cursor-pointer hover:shadow-sm"}
      `}
    >
      {/* Status indicator */}
      <span className="flex-shrink-0">
        {status === "complete"
          ? <CheckCircle size={15} className="text-green-600" />
          : status === "in_progress"
            ? <Circle size={15} className="text-blue-600 animate-pulse" />
            : status === "failed"
              ? <Circle size={15} className="text-red-400" />
              : <Circle size={15} className="text-gray-300" />
        }
      </span>

      {/* Icon */}
      <Icon size={14} className="flex-shrink-0 opacity-70" />

      {/* Label */}
      <span className="flex-1 leading-tight">{label}</span>

      {/* Chevron for completed */}
      {status === "complete" && !isActive && (
        <ChevronRight size={13} className="text-gray-300 flex-shrink-0" />
      )}
    </button>
  )
}

export default function ProgressSidebar({ onDeleteSession }) {
  const { session }         = useSession()
  const { currentStage, goToStage, getStageStatus } = usePipeline()

  const completedCount = STAGE_ORDER.filter(s => getStageStatus(s) === "complete").length
  const totalCount     = STAGE_ORDER.length
  const pct            = Math.round((completedCount / totalCount) * 100)

  return (
    <aside className="w-56 flex-shrink-0 bg-white border-r border-gray-100 flex flex-col overflow-hidden">
      {/* Goal + progress header */}
      <div className="p-4 border-b border-gray-100">
        {session?.goal?.description && (
          <p className="text-xs text-gray-400 leading-relaxed line-clamp-2 mb-3">
            {session.goal.description}
          </p>
        )}

        {/* Progress bar */}
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-[#1B3A5C] rounded-full transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="text-xs text-gray-400 flex-shrink-0">{pct}%</span>
        </div>
      </div>

      {/* Stage list */}
      <nav className="flex-1 overflow-y-auto p-3 space-y-1" aria-label="Pipeline stages">
        {STAGE_ORDER.map(stage => {
          const meta   = STAGE_LABELS[stage]
          const status = getStageStatus(stage)
          return (
            <StageItem
              key={stage}
              stage={stage}
              status={status}
              label={meta.label}
              iconName={meta.icon}
              isActive={stage === currentStage}
              onClick={() => goToStage(stage)}
            />
          )
        })}
      </nav>

      {/* Session actions */}
      {session && (
        <div className="p-3 border-t border-gray-100">
          <button
            onClick={onDeleteSession}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-400
                       hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
          >
            <Trash2 size={13} />
            Delete session
          </button>
        </div>
      )}
    </aside>
  )
}
