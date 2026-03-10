/**
 * SessionContext
 * Holds the active session.json state. Single source of truth for the UI.
 * All session data flows down through this context.
 */

import { createContext, useContext, useReducer, useCallback, useState } from "react"
import { api } from "../api"

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState = {
  sessions:        [],        // list of all sessions (for home screen)
  session:         null,      // currently loaded session object
  sessionId:       null,      // active session_id shorthand
  loading:         false,     // global loading flag
  error:           null       // global error message
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function sessionReducer(state, action) {
  switch (action.type) {

    case "SET_LOADING":
      return { ...state, loading: action.payload, error: null }

    case "SET_ERROR":
      return { ...state, loading: false, error: action.payload }

    case "CLEAR_ERROR":
      return { ...state, error: null }

    case "SET_SESSIONS":
      return { ...state, sessions: action.payload, loading: false }

    case "SET_SESSION":
      return {
        ...state,
        session:   action.payload,
        sessionId: action.payload?.session_id ?? null,
        loading:   false,
        error:     null
      }

    case "UPDATE_SESSION_CONFIG":
      if (!state.session) return state
      return {
        ...state,
        session: {
          ...state.session,
          config: { ...state.session.config, ...action.payload }
        }
      }

    case "UPDATE_STAGE_STATUS": {
      const { stage, status, result } = action.payload
      if (!state.session) return state
      return {
        ...state,
        session: {
          ...state.session,
          stages: {
            ...state.session.stages,
            [stage]: {
              ...(state.session.stages?.[stage] ?? {}),
              status,
              result: result ?? state.session.stages?.[stage]?.result
            }
          }
        }
      }
    }

    case "CLEAR_SESSION":
      return { ...state, session: null, sessionId: null, error: null }

    default:
      return state
  }
}

// ---------------------------------------------------------------------------
// Context + Provider
// ---------------------------------------------------------------------------

const SessionContext = createContext(null)

export function SessionProvider({ children }) {
  const [state, dispatch] = useReducer(sessionReducer, initialState)
  const [pendingFile, setPendingFile] = useState(null)

  // ---- Load all sessions (home screen) -----------------------------------
  const loadSessions = useCallback(async () => {
    dispatch({ type: "SET_LOADING", payload: true })
    try {
      const data = await api.listSessions()
      dispatch({ type: "SET_SESSIONS", payload: data })
    } catch (err) {
      dispatch({ type: "SET_ERROR", payload: "Could not reach the pipeline server. Is it running?" })
    }
  }, [])

  // ---- Create a new session -----------------------------------------------
  const createSession = useCallback(async (goal) => {
    dispatch({ type: "SET_LOADING", payload: true })
    try {
      const data = await api.createSession(goal)
      const session = data.session ?? data
      dispatch({ type: "SET_SESSION", payload: session })
      return session
    } catch (err) {
      dispatch({ type: "SET_ERROR", payload: "Failed to create a new session." })
      return null
    }
  }, [])

  // ---- Resume an existing session -----------------------------------------
  const resumeSession = useCallback(async (sessionId) => {
    dispatch({ type: "SET_LOADING", payload: true })
    try {
      const data = await api.loadSession(sessionId)
      const session = data.session ?? data
      dispatch({ type: "SET_SESSION", payload: session })
      return session
    } catch (err) {
      dispatch({ type: "SET_ERROR", payload: "Could not load the selected session." })
      return null
    }
  }, [])

  // ---- Refresh session from server ----------------------------------------
  const refreshSession = useCallback(async () => {
    if (!state.sessionId) return
    try {
      const data = await api.loadSession(state.sessionId)
      const session = data.session ?? data
      dispatch({ type: "SET_SESSION", payload: session })
    } catch (err) {
      // Silently ignore refresh errors
    }
  }, [state.sessionId])

  // ---- Delete session -------------------------------------------------------
  const deleteSession = useCallback(async (sessionId) => {
    try {
      await api.deleteSession(sessionId)
      dispatch({ type: "CLEAR_SESSION" })
      // Refresh the session list
      const data = await api.listSessions()
      dispatch({ type: "SET_SESSIONS", payload: data })
      return true
    } catch (err) {
      dispatch({ type: "SET_ERROR", payload: "Failed to delete the session." })
      return false
    }
  }, [])

  // ---- Local helpers -------------------------------------------------------
  const clearSession = useCallback(() => {
    dispatch({ type: "CLEAR_SESSION" })
    setPendingFile(null)
  }, [])

  const clearError = useCallback(() => {
    dispatch({ type: "CLEAR_ERROR" })
  }, [])

  const updateStageStatus = useCallback((stage, status, result) => {
    dispatch({ type: "UPDATE_STAGE_STATUS", payload: { stage, status, result } })
  }, [])

  // ---- Expose ---------------------------------------------------------------
  const value = {
    ...state,
    loadSessions,
    createSession,
    resumeSession,
    refreshSession,
    deleteSession,
    clearSession,
    clearError,
    updateStageStatus,
    pendingFile,
    setPendingFile
  }

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  )
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSession() {
  const ctx = useContext(SessionContext)
  if (!ctx) throw new Error("useSession must be used inside <SessionProvider>")
  return ctx
}
