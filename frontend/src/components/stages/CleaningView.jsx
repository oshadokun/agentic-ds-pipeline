import { useEffect, useState } from "react"
import { usePipeline } from "../../contexts/PipelineContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import DecisionCard from "../shared/DecisionCard"

export default function CleaningView() {
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus,
    decisions, setDecisions
  } = usePipeline()

  const [hasLoaded, setHasLoaded]     = useState(false)
  const [confirmed, setConfirmed]     = useState({})  // decisionId → true
  const [applying, setApplying]       = useState(false)

  const stageStatus = getStageStatus("cleaning")
  const isComplete  = stageStatus === "complete"
  const result      = stageResult

  useEffect(() => {
    async function load() {
      const r = isComplete
        ? await loadStageResult("cleaning")
        : await runStage("cleaning")          // first call — request decisions
      setHasLoaded(true)
    }
    load()
  }, []) // eslint-disable-line

  const required   = result?.decisions_required ?? []
  const needsDecisions = required.length > 0 && result?.status === "decisions_required"
  const allConfirmed   = required.length === 0 || required.every(d => confirmed[d.id])

  function handleConfirm(decisionId, value) {
    setDecisions({ [decisionId]: value })
    setConfirmed(c => ({ ...c, [decisionId]: true }))
  }

  async function applyDecisions() {
    setApplying(true)
    await runStage("cleaning", decisions)
    setApplying(false)
  }

  if (stageRunning || applying) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName={applying ? "Cleaning your data…" : "Scanning for data quality issues…"}
          message="This may take a moment."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Cleaning failed" message={stageError} />
      )}

      {/* Decisions phase */}
      {needsDecisions && hasLoaded && (
        <div className="space-y-4">
          <ExplanationPanel
            message={result.plain_english_summary ??
              "Here is what we found that needs attention. For each item, we have a recommendation — but you can choose a different approach if you prefer."}
          />

          {required.map(d => (
            <DecisionCard
              key={d.id}
              decision={d}
              onConfirm={handleConfirm}
            />
          ))}

          {allConfirmed && (
            <button
              onClick={applyDecisions}
              className="w-full py-3 rounded-xl bg-[#1B3A5C] text-white font-medium
                         hover:bg-[#162f4d] transition-colors text-sm"
            >
              Apply cleaning decisions
            </button>
          )}
        </div>
      )}

      {/* No decisions needed — run automatically */}
      {!needsDecisions && !isComplete && hasLoaded && !stageRunning && (
        <div className="text-center py-8">
          <p className="text-gray-500 text-sm mb-4">
            No manual decisions are needed. We'll apply automatic cleaning fixes now.
          </p>
          <button
            onClick={() => runStage("cleaning")}
            className="px-6 py-3 bg-[#1B3A5C] text-white rounded-xl font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Clean My Data
          </button>
        </div>
      )}

      {/* Complete — show summary */}
      {isComplete && result && (
        <div className="space-y-4">
          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {/* Before / after */}
          {result.rows_before !== undefined && (
            <div className="grid grid-cols-2 gap-3">
              <StatBox label="Rows before" value={result.rows_before?.toLocaleString()} />
              <StatBox label="Rows after"  value={result.rows_after?.toLocaleString()}  />
              <StatBox label="Columns before" value={result.cols_before} />
              <StatBox label="Columns after"  value={result.cols_after}  />
            </div>
          )}

          {/* Actions log — grouped by action type */}
          {result.actions_taken?.length > 0 && (
            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">What we did</p>
              <GroupedActionList actions={result.actions_taken} />
            </div>
          )}
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete}
        continueLabel="Continue to Prepare Features"
      />
    </div>
  )
}

// Groups cleaning action strings by their action-type prefix.
// Uses regex to extract "action verb phrase in 'ColumnName'" so varying
// numeric suffixes (e.g. "Capped 5 extreme values…" vs "Capped 12…") don't
// prevent grouping — the prefix is canonicalised to whatever precedes " in '".
function buildGroups(actions) {
  const texts  = actions.map(a => typeof a === "string" ? a : (a?.plain_english ?? String(a)))
  const groups = new Map()
  texts.forEach((text, i) => {
    const m = text.match(/^(.+) in '([^']+)'/)
    if (m) {
      const prefix = m[1]
      const col    = m[2]
      if (!groups.has(prefix)) groups.set(prefix, { cols: [], firstIndex: i })
      groups.get(prefix).cols.push(col)
    } else {
      groups.set(`__single__${i}`, { cols: [], firstIndex: i, single: text })
    }
  })
  return [...groups.entries()].sort((a, b) => a[1].firstIndex - b[1].firstIndex)
}

function GroupedActionList({ actions }) {
  const [expanded, setExpanded] = useState({})
  const sorted = buildGroups(actions)

  return (
    <ul className="space-y-1.5">
      {sorted.map(([key, { cols, single }], i) => {
        const isSingle = single !== undefined
        const isGroup  = !isSingle && cols.length > 1
        const shown    = cols.slice(0, 3).join(", ")
        const rest     = cols.length - 3
        const isExpanded = expanded[i]

        let lineText
        if (isSingle) {
          lineText = single
        } else if (cols.length === 1) {
          lineText = `${key} in '${cols[0]}'`
        } else {
          lineText = rest > 0
            ? `${key} in ${cols.length} columns (${shown}, and ${rest} more)`
            : `${key} in ${cols.length} columns (${shown})`
        }

        return (
          <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
            <span className="text-green-500 mt-0.5 flex-shrink-0">✓</span>
            <span className="flex-1">
              {lineText}
              {isGroup && rest > 0 && (
                <>
                  {" "}
                  <button
                    onClick={() => setExpanded(e => ({ ...e, [i]: !e[i] }))}
                    className="text-blue-500 hover:underline text-xs"
                  >
                    {isExpanded ? "show less" : "show all"}
                  </button>
                  {isExpanded && (
                    <ul className="mt-1.5 ml-1 space-y-0.5">
                      {cols.map((col, j) => (
                        <li key={j} className="text-xs text-gray-400">• {col}</li>
                      ))}
                    </ul>
                  )}
                </>
              )}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

function StatBox({ label, value }) {
  return (
    <div className="bg-gray-50 rounded-xl p-3 text-center">
      <p className="text-lg font-semibold text-gray-800">{value ?? "—"}</p>
      <p className="text-xs text-gray-400 mt-0.5">{label}</p>
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Clean Your Data</h2>
      <p className="text-gray-500 text-sm">
        We'll fix quality issues found during exploration — missing values, outliers, and
        inconsistencies. Nothing changes without your knowledge.
      </p>
    </div>
  )
}
