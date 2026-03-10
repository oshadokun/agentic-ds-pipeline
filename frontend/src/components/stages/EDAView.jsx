import { useEffect, useState } from "react"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { usePipeline } from "../../contexts/PipelineContext"
import { useSession } from "../../contexts/SessionContext"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import ExplanationPanel from "../shared/ExplanationPanel"
import StaticChart from "../charts/StaticChart"
import DecisionCard from "../shared/DecisionCard"

function buildPages(result) {
  if (!result?.findings) return []
  const { findings } = result
  const pages = []

  if (findings.target_analysis?.chart_path) {
    pages.push({
      key:       "target",
      title:     "Your target column — what we're trying to predict",
      caption:   findings.target_analysis.plain_english ?? "",
      chartPath: findings.target_analysis.chart_path
    })
  }

  if (findings.feature_analysis?.length > 0) {
    pages.push({
      key:       "features",
      title:     "How your columns are distributed",
      caption:   "Each bar shows how many rows fall into each range. Very skewed shapes or extreme spikes may need attention during cleaning.",
      chartPath: result.charts?.find(c => c.includes("feature_distributions")) ?? "",
      advisories: (findings.feature_analysis ?? []).filter(f => f.advisory).slice(0, 3)
    })
  }

  if (findings.correlations?.chart_path) {
    pages.push({
      key:        "correlations",
      title:      "Relationships between your columns",
      caption:    findings.correlations.plain_english ?? "",
      chartPath:  findings.correlations.chart_path,
      highCorr:   findings.correlations.high_corr_pairs ?? []
    })
  }

  if (findings.feature_vs_target?.chart_path) {
    pages.push({
      key:      "vs_target",
      title:    "How columns vary across your outcomes",
      caption:  (findings.feature_vs_target.insights ?? []).join(" ") ||
                "Columns whose distributions differ across outcomes are likely useful predictors.",
      chartPath: findings.feature_vs_target.chart_path
    })
  }

  return pages
}

export default function EDAView() {
  const { sessionId, session } = useSession()
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus
  } = usePipeline()

  const [pageIndex, setPageIndex]   = useState(0)
  const [hasLoaded, setHasLoaded]   = useState(false)
  const [pendingDecisions, setPendingDecisions] = useState({})

  const stageStatus       = getStageStatus("eda")
  const isComplete        = stageStatus === "complete"
  const result            = stageResult
  const tsColumns         = session?.config?.time_series_columns ?? []
  // Never show identifier decision cards for date/time columns
  const decisionsRequired = (result?.decisions_required ?? []).filter(
    d => !d.id.startsWith("id_col__") || !tsColumns.includes(d.id.replace("id_col__", ""))
  )
  const needsDecisions    = decisionsRequired.length > 0
  const allDecided        = decisionsRequired.every(d => pendingDecisions[d.id] !== undefined)

  function handleDecision(id, value) {
    setPendingDecisions(prev => ({ ...prev, [id]: value }))
  }

  function submitDecisions() {
    runStage("eda", pendingDecisions)
  }

  useEffect(() => {
    async function load() {
      if (isComplete) await loadStageResult("eda")
      setHasLoaded(true)
    }
    load()
  }, []) // eslint-disable-line

  const pages = buildPages(result)
  const page  = pages[pageIndex] ?? null

  if (stageRunning) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName="Exploring your data…"
          message="Analysing distributions, correlations, and patterns. This may take up to a minute."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StageHeader />

      {stageError && (
        <AlertBanner type="error" title="Analysis failed" message={stageError} />
      )}

      {!isComplete && hasLoaded && (
        <div className="text-center py-8">
          <p className="text-gray-500 text-sm mb-4">
            We'll analyse every column in your dataset and show you what we find —
            distributions, relationships, and anything that will affect modelling.
            We'll walk through it one chart at a time.
          </p>
          <button
            onClick={() => runStage("eda")}
            className="px-6 py-3 bg-[#1B3A5C] text-white rounded-xl font-medium
                       hover:bg-[#162f4d] transition-colors"
          >
            Explore My Data
          </button>
        </div>
      )}

      {isComplete && pages.length > 0 && (
        <div className="space-y-5">
          {/* Orchestrator summary — shown on first page */}
          {result?.plain_english_summary && pageIndex === 0 && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {/* Page dots */}
          <div className="flex items-center justify-between text-sm text-gray-400">
            <span>Chart {pageIndex + 1} of {pages.length}</span>
            <div className="flex gap-1.5">
              {pages.map((p, i) => (
                <button
                  key={p.key}
                  onClick={() => setPageIndex(i)}
                  className={`w-2 h-2 rounded-full transition-colors ${
                    i === pageIndex ? "bg-[#1B3A5C]" : "bg-gray-200 hover:bg-gray-300"
                  }`}
                  aria-label={`Chart ${i + 1}`}
                />
              ))}
            </div>
          </div>

          {/* Current chart card */}
          {page && (
            <div className="border border-gray-100 rounded-2xl p-5 bg-white shadow-sm">
              <h3 className="font-semibold text-gray-800 mb-4">{page.title}</h3>
              {page.chartPath
                ? <StaticChart sessionId={sessionId} chartPath={page.chartPath} caption={page.caption} />
                : <p className="text-sm text-gray-400 text-center py-8">Chart not available.</p>
              }
            </div>
          )}

          {/* Page-specific advisories */}
          {page?.advisories?.length > 0 && (
            <div className="space-y-2">
              {page.advisories.map((f, i) => (
                <AlertBanner key={i} type="info" message={f.advisory} />
              ))}
            </div>
          )}

          {page?.highCorr?.length > 0 && (
            <AlertBanner
              type="warning"
              title="Highly correlated columns found"
              message={`${page.highCorr.length} pair${page.highCorr.length > 1 ? "s" : ""} of columns are very similar to each other. We'll handle these in feature engineering.`}
            />
          )}

          {/* Chart navigation */}
          <div className="flex gap-3">
            <button
              onClick={() => setPageIndex(i => Math.max(0, i - 1))}
              disabled={pageIndex === 0}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl border border-gray-200
                         text-sm text-gray-600 disabled:opacity-30 hover:bg-gray-50 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" /> Previous
            </button>
            <button
              onClick={() => setPageIndex(i => Math.min(pages.length - 1, i + 1))}
              disabled={pageIndex === pages.length - 1}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl border border-gray-200
                         text-sm text-gray-600 disabled:opacity-30 hover:bg-gray-50 transition-colors ml-auto"
            >
              Next <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* Overview stats strip */}
          {result?.findings?.overview && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: "Rows",    value: result.findings.overview.row_count?.toLocaleString() },
                { label: "Columns", value: result.findings.overview.column_count },
                { label: "Numeric columns", value: result.findings.overview.numeric_cols?.length },
                { label: "Missing values",  value: `${((result.findings.overview.missing_pct ?? 0) * 100).toFixed(1)}%` }
              ].map(({ label, value }) => (
                <div key={label} className="bg-gray-50 rounded-xl p-3 text-center">
                  <p className="text-lg font-semibold text-gray-800">{value ?? "—"}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{label}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Decisions panel — shown after first analysis run */}
      {isComplete === false && needsDecisions && (
        <div className="space-y-4">
          <div className="border-t border-gray-100 pt-4">
            <h3 className="font-semibold text-gray-800 mb-1">We need your input before continuing</h3>
            <p className="text-sm text-gray-500 mb-4">
              We found a few things that need a decision from you.
              Review each one and choose how you'd like to proceed.
            </p>
            {decisionsRequired.map(d => (
              <DecisionCard
                key={d.id}
                decision={d}
                onConfirm={handleDecision}
                disabled={stageRunning}
              />
            ))}
          </div>
          <button
            onClick={submitDecisions}
            disabled={!allDecided || stageRunning}
            className="w-full py-3 rounded-xl bg-[#1B3A5C] text-white font-medium text-sm
                       disabled:opacity-40 hover:bg-[#162f4d] transition-colors"
          >
            {stageRunning ? "Applying your choices…" : "Confirm and continue"}
          </button>
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!isComplete}
        continueLabel="Continue to Clean Data"
      />
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Explore Your Data</h2>
      <p className="text-gray-500 text-sm">
        Before making any changes, let's understand what your data looks like.
        We'll walk through each finding one chart at a time.
      </p>
    </div>
  )
}
