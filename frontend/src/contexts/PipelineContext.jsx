/**
 * PipelineContext
 * Controls the stage flow — which stage is active, how to advance,
 * how to run a stage, and how to collect decisions.
 * Works in tandem with SessionContext which holds the session data.
 */

import { createContext, useContext, useReducer, useCallback } from "react"
import { api } from "../api"
import { useSession } from "./SessionContext"
import { STAGE_ORDER } from "./stageConfig"

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState = {
  currentStage:    null,        // currently visible stage key
  stageRunning:    false,       // an agent is currently executing
  stageResult:     null,        // latest result from the backend
  decisions:       {},          // decisions collected for the current run
  stageError:      null         // error from the most recent stage run
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function pipelineReducer(state, action) {
  switch (action.type) {

    case "SET_STAGE":
      return {
        ...state,
        currentStage: action.payload,
        stageResult:  null,
        stageError:   null,
        decisions:    {}
      }

    case "STAGE_RUNNING":
      return { ...state, stageRunning: true, stageError: null }

    case "STAGE_DONE":
      return {
        ...state,
        stageRunning: false,
        stageResult:  action.payload,
        stageError:   null
      }

    case "STAGE_ERROR":
      return { ...state, stageRunning: false, stageError: action.payload }

    case "SET_STAGE_RESULT":
      return { ...state, stageResult: action.payload }

    case "SET_DECISIONS":
      return { ...state, decisions: { ...state.decisions, ...action.payload } }

    case "CLEAR_DECISIONS":
      return { ...state, decisions: {} }

    case "CLEAR_STAGE_ERROR":
      return { ...state, stageError: null }

    default:
      return state
  }
}

// ---------------------------------------------------------------------------
// Context + Provider
// ---------------------------------------------------------------------------

const PipelineContext = createContext(null)

export function PipelineProvider({ children }) {
  const [state, dispatch] = useReducer(pipelineReducer, initialState)
  const { sessionId, session, updateStageStatus, refreshSession } = useSession()

  // ---- Navigate to a stage ------------------------------------------------
  const goToStage = useCallback((stage) => {
    dispatch({ type: "SET_STAGE", payload: stage })
  }, [])

  // ---- Advance to next stage -----------------------------------------------
  const goToNextStage = useCallback(() => {
    const idx  = STAGE_ORDER.indexOf(state.currentStage)
    const next = STAGE_ORDER[idx + 1]
    if (next) dispatch({ type: "SET_STAGE", payload: next })
  }, [state.currentStage])

  // ---- Go back one stage ---------------------------------------------------
  const goToPrevStage = useCallback(() => {
    const idx  = STAGE_ORDER.indexOf(state.currentStage)
    const prev = STAGE_ORDER[idx - 1]
    if (prev) dispatch({ type: "SET_STAGE", payload: prev })
  }, [state.currentStage])

  // ---- Run a pipeline stage -----------------------------------------------
  const runStage = useCallback(async (stage, decisions = {}) => {
    if (!sessionId) return null
    dispatch({ type: "STAGE_RUNNING" })
    updateStageStatus(stage, "in_progress")

    try {
      // Auto-fill recommendations for any decisions the user didn't explicitly set,
      // so the backend never receives an empty dict when it's expecting phase-2 input.
      const pending = state.stageResult?.decisions_required ?? []
      const filledDecisions = { ...decisions }
      pending.forEach(d => {
        if (!(d.id in filledDecisions) && d.recommendation !== undefined) {
          filledDecisions[d.id] = d.recommendation
        }
      })
      const effectiveDecisions = Object.keys(filledDecisions).length > 0 ? filledDecisions : decisions

      const result = await api.runStage(sessionId, stage, effectiveDecisions)

      if (result.status === "failed") {
        dispatch({ type: "STAGE_ERROR", payload: result.plain_english_summary })
        updateStageStatus(stage, "failed")
        return result
      }

      dispatch({ type: "STAGE_DONE", payload: result })
      updateStageStatus(stage, result.status === "decisions_required" ? "in_progress" : "complete", result)
      await refreshSession()
      return result

    } catch (err) {
      const msg = "Something went wrong running this stage. Please try again."
      dispatch({ type: "STAGE_ERROR", payload: msg })
      updateStageStatus(stage, "failed")
      return null
    }
  }, [sessionId, updateStageStatus, refreshSession])

  // ---- Fetch a previously stored stage result ------------------------------
  const loadStageResult = useCallback(async (stage) => {
    if (!sessionId) return null
    try {
      const result = await api.getStageResult(sessionId, stage)
      dispatch({ type: "SET_STAGE_RESULT", payload: result })
      return result
    } catch {
      return null
    }
  }, [sessionId])

  // ---- Merge decisions into current set ------------------------------------
  const setDecisions = useCallback((d) => {
    dispatch({ type: "SET_DECISIONS", payload: d })
  }, [])

  const clearDecisions = useCallback(() => {
    dispatch({ type: "CLEAR_DECISIONS" })
  }, [])

  const clearStageError = useCallback(() => {
    dispatch({ type: "CLEAR_STAGE_ERROR" })
  }, [])

  // ---- Derive stage statuses from session ---------------------------------
  const getStageStatus = useCallback((stage) => {
    return session?.stages?.[stage]?.status ?? "pending"
  }, [session])

  // ---- Current stage index for progress calculation -----------------------
  const currentStageIndex = STAGE_ORDER.indexOf(state.currentStage)
  const progressPct = currentStageIndex >= 0
    ? Math.round(((currentStageIndex) / STAGE_ORDER.length) * 100)
    : 0

  const value = {
    ...state,
    currentStageIndex,
    progressPct,
    goToStage,
    goToNextStage,
    goToPrevStage,
    runStage,
    loadStageResult,
    setDecisions,
    clearDecisions,
    clearStageError,
    getStageStatus
  }

  return (
    <PipelineContext.Provider value={value}>
      {children}
    </PipelineContext.Provider>
  )
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function usePipeline() {
  const ctx = useContext(PipelineContext)
  if (!ctx) throw new Error("usePipeline must be used inside <PipelineProvider>")
  return ctx
}
