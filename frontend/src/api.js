/**
 * api.js — Centralised API client for the pipeline management backend (port 8001).
 * This is the ONLY place that knows the backend URL.
 * Do not use fetch() directly anywhere else in the frontend.
 *
 * Note: the deployed model API on port 8000 is a different service — it is NOT
 * accessed through this client. It is accessed directly by the user after deployment.
 */

const BASE_URL = "/api"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`)
  if (!res.ok) await _throwApiError(res)
  return res.json()
}

async function post(path, body = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body)
  })
  if (!res.ok) await _throwApiError(res)
  return res.json()
}

async function patch(path, body = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method:  "PATCH",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body)
  })
  if (!res.ok) await _throwApiError(res)
  return res.json()
}

async function del(path) {
  const res = await fetch(`${BASE_URL}${path}`, { method: "DELETE" })
  if (!res.ok) await _throwApiError(res)
  return res.json()
}

async function upload(path, file) {
  const form = new FormData()
  form.append("file", file)
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    body:   form
    // No Content-Type header — browser sets it with boundary
  })
  if (!res.ok) await _throwApiError(res)
  return res.json()
}

async function _throwApiError(res) {
  let detail = `Server error (${res.status})`
  try {
    const body = await res.json()
    detail = body.detail ?? body.message ?? detail
  } catch {
    // Could not parse error body — use the status code message
  }
  throw new Error(detail)
}

// ---------------------------------------------------------------------------
// Public API surface
// ---------------------------------------------------------------------------

export const api = {

  // -- Session management ---------------------------------------------------

  /** List all sessions for the home screen */
  listSessions: () =>
    get("/sessions"),

  /** Create a new session with the given goal description */
  createSession: (goal) =>
    post("/sessions", { goal }),

  /** Load a session by ID */
  loadSession: (sessionId) =>
    get(`/sessions/${sessionId}`),

  /** Delete a session (requires confirm=true param) */
  deleteSession: (sessionId) =>
    del(`/sessions/${sessionId}?confirm=true`),

  /** Update the goal for an existing session */
  updateGoal: (sessionId, goalUpdates) =>
    patch(`/sessions/${sessionId}/goal`, goalUpdates),

  // -- Data upload ----------------------------------------------------------

  /** Upload a CSV file to a session */
  uploadFile: (sessionId, file) =>
    upload(`/sessions/${sessionId}/data`, file),

  // -- Privacy --------------------------------------------------------------

  /** Submit privacy decisions for a session */
  submitPrivacyDecisions: (sessionId, decisions) =>
    post(`/sessions/${sessionId}/privacy`, { decisions }),

  // -- Pipeline execution ---------------------------------------------------

  /**
   * Run a pipeline stage.
   * @param {string} sessionId
   * @param {string} stage - e.g. "training", "evaluation"
   * @param {object} decisions - key/value pairs from user decisions
   */
  runStage: (sessionId, stage, decisions = {}) =>
    post(`/sessions/${sessionId}/stages/${stage}/run`, { decisions }),

  /** Get the stored result for a completed stage */
  getStageResult: (sessionId, stage) =>
    get(`/sessions/${sessionId}/stages/${stage}/result`),

  // -- Charts ---------------------------------------------------------------

  /**
   * Fetch a chart PNG as a Blob.
   * @param {string} sessionId
   * @param {string} chartPath - file path returned by the agent (e.g. sessions/xxx/reports/eda/target.png)
   */
  getChart: async (sessionId, chartPath) => {
    const url = `/api/sessions/${sessionId}/charts?path=${encodeURIComponent(chartPath)}`
    const res = await fetch(url)
    if (!res.ok) throw new Error(`Chart not found: ${chartPath}`)
    return res.blob()
  },

  /** Get chart as an object URL string (convenience wrapper) */
  getChartUrl: async (sessionId, chartPath) => {
    const blob = await api.getChart(sessionId, chartPath)
    return URL.createObjectURL(blob)
  },

  // -- Report ---------------------------------------------------------------

  /** Get the assembled final report */
  getReport: (sessionId) =>
    get(`/sessions/${sessionId}/report`),

  /** Get the URL to download the HTML report (use as window.location.href) */
  reportDownloadUrl: (sessionId) =>
    `/api/sessions/${sessionId}/report/html`,

  /** Get the generated API code (app.py content) */
  getApiCode: (sessionId) =>
    get(`/sessions/${sessionId}/code/api`),

  /** Download URLs for analysis script / notebook */
  scriptDownloadUrl:   (sessionId) => `/api/sessions/${sessionId}/code/script`,
  notebookDownloadUrl: (sessionId) => `/api/sessions/${sessionId}/code/notebook`,

  // -- Health ---------------------------------------------------------------

  /** Check that the backend is running */
  health: () =>
    get("/health")
}
