/**
 * App.jsx
 * Root component. Manages:
 *   - Home screen (session list / new session)
 *   - Active session shell + stage routing
 *
 * All data access flows through SessionContext and PipelineContext.
 */

import { useEffect, lazy, Suspense } from "react"
import { SessionProvider, useSession } from "./contexts/SessionContext"
import { PipelineProvider, usePipeline } from "./contexts/PipelineContext"
import { STAGE_ORDER, STAGE_LABELS } from "./contexts/stageConfig"
import AppShell from "./components/shell/AppShell"
import AgentRunning from "./components/shared/AgentRunning"
import AlertBanner from "./components/shared/AlertBanner"
import { Plus, ChevronRight, Brain } from "lucide-react"

// ---------------------------------------------------------------------------
// Lazy-load stage views (reduces initial bundle size)
// ---------------------------------------------------------------------------
const GoalCaptureView        = lazy(() => import("./components/stages/GoalCaptureView"))
const IngestionView          = lazy(() => import("./components/stages/IngestionView"))
const ValidationView         = lazy(() => import("./components/stages/ValidationView"))
const EDAView                = lazy(() => import("./components/stages/EDAView"))
const CleaningView           = lazy(() => import("./components/stages/CleaningView"))
const FeatureEngineeringView = lazy(() => import("./components/stages/FeatureEngineeringView"))
const NormalisationView      = lazy(() => import("./components/stages/NormalisationView"))
const SplittingView          = lazy(() => import("./components/stages/SplittingView"))
const TrainingView           = lazy(() => import("./components/stages/TrainingView"))
const EvaluationView         = lazy(() => import("./components/stages/EvaluationView"))
const TuningView             = lazy(() => import("./components/stages/TuningView"))
const ExplainabilityView     = lazy(() => import("./components/stages/ExplainabilityView"))
const DeploymentView         = lazy(() => import("./components/stages/DeploymentView"))
const MonitoringView         = lazy(() => import("./components/stages/MonitoringView"))

const STAGE_VIEWS = {
  ingestion:           IngestionView,
  validation:          ValidationView,
  eda:                 EDAView,
  cleaning:            CleaningView,
  feature_engineering: FeatureEngineeringView,
  normalisation:       NormalisationView,
  splitting:           SplittingView,
  training:            TrainingView,
  evaluation:          EvaluationView,
  tuning:              TuningView,
  explainability:      ExplainabilityView,
  deployment:          DeploymentView,
  monitoring:          MonitoringView
}

// ---------------------------------------------------------------------------
// Home screen — shown when no session is active
// ---------------------------------------------------------------------------

function HomeScreen() {
  const { sessions, loading, error, loadSessions, createSession, resumeSession, clearError } = useSession()
  const { goToStage } = usePipeline()

  useEffect(() => { loadSessions() }, [loadSessions])

  async function handleNew() {
    goToStage("__goal__")
  }

  async function handleResume(sessionId) {
    const session = await resumeSession(sessionId)
    if (!session) return
    // Navigate to the last active stage
    const lastStage = session.stages
      ? STAGE_ORDER.slice().reverse().find(s => session.stages[s]?.status === "complete")
      : null
    goToStage(lastStage ?? "ingestion")
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-8">
      <div className="max-w-xl w-full">

        {/* Hero */}
        <div className="mb-10 text-center">
          <div className="inline-flex w-14 h-14 rounded-2xl bg-[#1B3A5C]
                          items-center justify-center mb-5">
            <Brain size={28} className="text-white" />
          </div>
          <h1 className="text-3xl font-serif text-gray-900 mb-3">
            Data Science Pipeline
          </h1>
          <p className="text-gray-500 leading-relaxed">
            Build, train, and deploy a machine learning model —
            step by step, in plain English.
          </p>
        </div>

        {/* Error */}
        {error && (
          <AlertBanner
            type="error"
            message={error}
            onClose={clearError}
            className="mb-4"
          />
        )}

        {/* Start new */}
        <button
          onClick={handleNew}
          className="w-full mb-6 py-4 px-6 bg-[#1B3A5C] text-white rounded-2xl
                     font-medium hover:bg-[#2E6099] transition-colors
                     flex items-center justify-center gap-3 text-base shadow-md"
        >
          <Plus size={20} />
          Start a New Pipeline
        </button>

        {/* Previous sessions */}
        {loading && (
          <p className="text-center text-sm text-gray-400">Loading sessions…</p>
        )}

        {!loading && sessions.length > 0 && (
          <>
            <div className="flex items-center gap-3 mb-4">
              <div className="flex-1 h-px bg-gray-200" />
              <span className="text-sm text-gray-400 whitespace-nowrap">
                or continue a previous session
              </span>
              <div className="flex-1 h-px bg-gray-200" />
            </div>

            <div className="space-y-3">
              {sessions.map(s => (
                <button
                  key={s.session_id}
                  onClick={() => handleResume(s.session_id)}
                  className="w-full p-4 bg-white rounded-xl border border-gray-200
                             hover:border-blue-300 hover:shadow-md transition-all
                             text-left flex items-start justify-between gap-4"
                >
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 truncate text-sm">
                      {s.goal || "No goal recorded"}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      Last updated {new Date(s.last_updated).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="text-xs px-2 py-1 bg-blue-50 text-blue-700
                                     rounded-full font-medium">
                      {STAGE_LABELS[s.last_stage]?.label ?? s.last_stage ?? "Not started"}
                    </span>
                    <ChevronRight size={15} className="text-gray-300" />
                  </div>
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Stage router — renders the active stage view
// ---------------------------------------------------------------------------

function StageRouter() {
  const { currentStage, stageRunning } = usePipeline()

  if (currentStage === "__goal__") {
    return (
      <Suspense fallback={<AgentRunning stageName="Loading…" />}>
        <GoalCaptureView />
      </Suspense>
    )
  }

  const View = STAGE_VIEWS[currentStage]

  if (!View) {
    return (
      <div className="py-16 text-center text-gray-400">
        Select a stage from the sidebar to begin.
      </div>
    )
  }

  return (
    <Suspense fallback={<AgentRunning stageName={STAGE_LABELS[currentStage]?.label ?? "Loading…"} />}>
      <View />
    </Suspense>
  )
}

// ---------------------------------------------------------------------------
// App root with context providers
// ---------------------------------------------------------------------------

function AppContent() {
  const { session } = useSession()
  const { currentStage } = usePipeline()

  // No active session or navigating to goal capture → home or goal screen
  if (!session && currentStage !== "__goal__") {
    return <HomeScreen />
  }

  // Goal capture — no shell yet
  if (currentStage === "__goal__") {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-8">
        <div className="max-w-xl w-full">
          <Suspense fallback={<AgentRunning stageName="Loading…" />}>
            <GoalCaptureView />
          </Suspense>
        </div>
      </div>
    )
  }

  // Active session with pipeline shell
  return (
    <AppShell>
      <StageRouter />
    </AppShell>
  )
}

export default function App() {
  return (
    <SessionProvider>
      <PipelineProvider>
        <AppContent />
      </PipelineProvider>
    </SessionProvider>
  )
}
