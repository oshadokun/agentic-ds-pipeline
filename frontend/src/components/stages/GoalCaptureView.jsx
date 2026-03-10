/**
 * GoalCaptureView
 * Upload file → analyse columns → show ONE best goal suggestion → populate text box.
 * User can edit freely. "Begin Pipeline" activates as soon as any goal text is present.
 * State is entirely local — resets cleanly on every new render.
 */

import { useState, useRef } from "react"
import { Brain, ArrowRight, Upload, Sparkles } from "lucide-react"
import { useSession } from "../../contexts/SessionContext"
import { usePipeline } from "../../contexts/PipelineContext"
import AlertBanner from "../shared/AlertBanner"


// ---------------------------------------------------------------------------
// Client-side CSV analysis — reads only headers + first 10 rows, no upload
// ---------------------------------------------------------------------------

function parseCSVLine(line) {
  const result = []
  let current = "", inQuotes = false
  for (const ch of line) {
    if (ch === '"') { inQuotes = !inQuotes }
    else if (ch === "," && !inQuotes) { result.push(current.trim()); current = "" }
    else { current += ch }
  }
  result.push(current.trim())
  return result
}

function detectColType(values) {
  const nonEmpty = values.filter(v => v && v.trim() !== "")
  if (!nonEmpty.length) return "unknown"
  const unique = [...new Set(nonEmpty.map(v => v.toLowerCase().trim()))]
  if (unique.length <= 2) return "binary"
  if (nonEmpty.every(v => !isNaN(parseFloat(v)) && v.trim() !== "")) return "numeric"
  const dateRe = /\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}/
  if (nonEmpty.filter(v => dateRe.test(v)).length / nonEmpty.length > 0.6) return "date"
  if (unique.length <= 10) return "categorical"
  return "text"
}

function humanise(col) {
  return col.replace(/_/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase()
}

const OHLCV_COLS = ["open", "high", "low", "close", "volume"]

/** Returns exactly ONE suggestion — the single best goal for this dataset. */
function bestSuggestion(csvText) {
  const lines = csvText.split(/\r?\n/).filter(l => l.trim()).slice(0, 11)
  if (lines.length < 2) return null

  const headers = parseCSVLine(lines[0])
  const rows    = lines.slice(1).map(parseCSVLine)

  const types = {}
  headers.forEach((col, i) => {
    types[col] = detectColType(rows.map(r => r[i] ?? ""))
  })

  const headerNorms = headers.map(h => h.toLowerCase().trim())
  const dateCol = headers.find(c => types[c] === "date")
  const numCols = headers.filter(c => types[c] === "numeric")
  const binCols = headers.filter(c => types[c] === "binary")
  const catCols = headers.filter(c => types[c] === "categorical")

  // 1. OHLCV financial data → always predict Close
  if (OHLCV_COLS.every(col => headerNorms.includes(col))) {
    const closeCol = headers.find(h => h.toLowerCase().trim() === "close")
    return {
      goal: `I want to predict the ${humanise(closeCol)} price based on the other columns in my data.`,
      why:  `Your file looks like stock market data (Open, High, Low, Close, Volume). Predicting Close price is the standard goal for this type of dataset.`
    }
  }

  // 2. Date + numeric → time series forecast
  if (dateCol && numCols.length > 0) {
    const target = numCols[numCols.length - 1]
    return {
      goal: `I want to forecast ${humanise(target)} over time using patterns from my historical data.`,
      why:  `Your file has a date column and numerical values — this is a classic setup for forecasting future values.`
    }
  }

  // 3. Binary column → classification
  if (binCols.length > 0) {
    const target = binCols[binCols.length - 1]
    return {
      goal: `I want to predict whether ${humanise(target)} will happen, based on everything else in my data.`,
      why:  `"${humanise(target)}" has two possible outcomes — we can train a model to predict which is more likely for each new row.`
    }
  }

  // 4. Multiple numeric columns (no date) → regression
  if (numCols.length > 1) {
    const target = numCols[numCols.length - 1]
    return {
      goal: `I want to estimate the ${humanise(target)} for new entries using the other information in my data.`,
      why:  `"${humanise(target)}" is a number — we can train a model to predict its value for records we haven't seen yet.`
    }
  }

  // 5. Categorical column → multiclass classification
  if (catCols.length > 0) {
    const target = catCols[catCols.length - 1]
    return {
      goal: `I want to predict which ${humanise(target)} applies to a new entry, based on the rest of my data.`,
      why:  `"${humanise(target)}" has a small set of possible values — we can build a model that assigns the right one.`
    }
  }

  // 6. Fallback
  return {
    goal: `I want to find patterns in my data and use them to predict an outcome for new entries.`,
    why:  `We'll walk you through choosing exactly what to predict once your file is loaded into the pipeline.`
  }
}


// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function GoalCaptureView() {
  const { createSession, setPendingFile, error, clearError } = useSession()
  const { goToStage } = usePipeline()

  // All state is local — resets cleanly on every mount (new session = new render)
  const [goal, setGoal]           = useState("")
  const [loading, setLoading]     = useState(false)
  const [suggestion, setSuggestion] = useState(null)   // { goal, why } | null
  const [fileName, setFileName]   = useState(null)
  const fileInputRef              = useRef(null)

  const canSubmit    = goal.trim().length > 0
  const hasFile      = fileName !== null

  async function handleSubmit() {
    if (!canSubmit || loading) return
    setLoading(true)
    const session = await createSession(goal.trim())
    setLoading(false)
    if (session) goToStage("ingestion")
  }

  function handleFile(file) {
    if (!file?.name.toLowerCase().endsWith(".csv")) return
    setPendingFile(file)
    setFileName(file.name)
    const reader = new FileReader()
    reader.onload = e => {
      const sug = bestSuggestion(e.target.result)
      setSuggestion(sug)
      if (sug) setGoal(sug.goal)
    }
    reader.readAsText(file)
  }

  function resetFile() {
    setPendingFile(null)
    setFileName(null)
    setSuggestion(null)
    setGoal("")
    clearError?.()
  }

  return (
    <div className="animate-fade-in space-y-5">

      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-[#1B3A5C] flex items-center justify-center flex-shrink-0">
          <Brain size={20} className="text-white" />
        </div>
        <div>
          <h1 className="text-2xl font-serif text-gray-900">What would you like to predict?</h1>
          <p className="text-gray-500 text-sm mt-0.5">
            Upload your data file and we'll suggest a goal based on what we find.
          </p>
        </div>
      </div>

      {error && <AlertBanner type="error" message={error} onClose={clearError} />}

      {/* File upload — shown until a file is loaded */}
      {!hasFile && (
        <>
          <div
            className="border-2 border-dashed border-gray-200 rounded-2xl p-7 text-center
                       cursor-pointer hover:border-blue-300 hover:bg-gray-50 transition-colors"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === "Enter" && fileInputRef.current?.click()}
          >
            <Upload className="w-9 h-9 text-gray-300 mx-auto mb-2" />
            <p className="font-medium text-gray-700 text-sm mb-0.5">Drop your CSV file here</p>
            <p className="text-gray-400 text-xs">or click to browse — we'll read the columns and suggest a goal</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={e => handleFile(e.target.files[0])}
            />
          </div>

          <div className="flex items-center gap-3">
            <hr className="flex-1 border-gray-200" />
            <span className="text-xs text-gray-400 whitespace-nowrap">or describe your goal yourself</span>
            <hr className="flex-1 border-gray-200" />
          </div>
        </>
      )}

      {/* Suggestion banner — shown after file is analysed */}
      {hasFile && suggestion && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
          <div className="flex items-center gap-1.5 mb-2">
            <Sparkles size={13} className="text-amber-500" />
            <p className="text-xs font-semibold text-amber-700 uppercase tracking-wide">
              Suggested goal based on "{fileName}"
            </p>
          </div>
          <p className="text-xs text-amber-700 leading-relaxed">{suggestion.why}</p>
          <button
            onClick={resetFile}
            className="mt-2 text-xs text-amber-600 hover:text-amber-800 underline transition-colors"
          >
            Upload a different file
          </button>
        </div>
      )}

      {/* Goal textarea */}
      <div>
        {hasFile && (
          <p className="text-xs text-gray-400 mb-1.5">
            Edit the goal below if you'd like something different:
          </p>
        )}
        <textarea
          value={goal}
          onChange={e => { setGoal(e.target.value); clearError?.() }}
          onKeyDown={e => e.key === "Enter" && e.metaKey && handleSubmit()}
          placeholder="e.g. I want to predict which customers are likely to cancel their subscription…"
          rows={3}
          className={`w-full px-4 py-3 rounded-xl border text-sm text-gray-800
                      resize-none focus:outline-none focus:ring-2 focus:ring-blue-400
                      transition-shadow leading-relaxed bg-white
                      ${error ? "border-red-300" : "border-gray-200"}`}
        />
      </div>

      {/* Begin Pipeline */}
      <div className="flex justify-end">
        <button
          onClick={handleSubmit}
          disabled={!canSubmit || loading}
          className="flex items-center gap-2 px-6 py-2.5 bg-[#1B3A5C] text-white
                     rounded-xl text-sm font-medium hover:bg-[#2E6099]
                     disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Creating…" : "Begin Pipeline"}
          {!loading && <ArrowRight size={16} />}
        </button>
      </div>

    </div>
  )
}
