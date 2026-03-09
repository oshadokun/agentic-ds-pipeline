import { useState, useRef, useEffect } from "react"
import { Upload, CheckCircle, Lock } from "lucide-react"
import { usePipeline } from "../../contexts/PipelineContext"
import { useSession } from "../../contexts/SessionContext"
import { api } from "../../api"
import AgentRunning from "../shared/AgentRunning"
import AlertBanner from "../shared/AlertBanner"
import StageNavigation from "../shell/StageNavigation"
import DataPreviewTable from "../shared/DataPreviewTable"
import ExplanationPanel from "../shared/ExplanationPanel"

export default function IngestionView() {
  const { session, sessionId, refreshSession } = useSession()
  const {
    runStage, loadStageResult,
    stageRunning, stageResult, stageError,
    goToNextStage, getStageStatus
  } = usePipeline()

  const [uploading, setUploading]         = useState(false)
  const [uploadError, setUploadError]     = useState(null)
  const [dragOver, setDragOver]           = useState(false)
  const [privacyChoices, setPrivacyChoices] = useState({})
  const [privacySubmitting, setPrivacySubmitting] = useState(false)
  const [targetColumn, setTargetColumn]   = useState(session?.goal?.target_column ?? "")
  const [taskType, setTaskType]           = useState(session?.goal?.task_type ?? "")
  const [goalSaving, setGoalSaving]       = useState(false)
  const fileInputRef = useRef(null)

  const stageStatus       = getStageStatus("ingestion")
  const isComplete        = stageStatus === "complete"
  const privacyAck        = session?.privacy?.user_acknowledged
  const result            = stageResult
  const columns           = result?.structural_check?.column_names ?? []
  const columnsToCheck    = result?.privacy?.columns_to_check
                          ?? session?.privacy?.columns_to_check
                          ?? []
  const goalConfirmed     = !!(targetColumn && taskType)

  useEffect(() => {
    if (isComplete) loadStageResult("ingestion")
  }, []) // eslint-disable-line

  async function handleFile(file) {
    if (!file?.name.toLowerCase().endsWith(".csv")) {
      setUploadError("Only CSV files are supported at this time.")
      return
    }
    setUploadError(null)
    setUploading(true)
    try {
      await api.uploadFile(sessionId, file)
      await runStage("ingestion")
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setUploading(false)
    }
  }

  async function submitPrivacy() {
    setPrivacySubmitting(true)
    try {
      await api.submitPrivacyDecisions(sessionId, privacyChoices)
      await refreshSession()
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setPrivacySubmitting(false)
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  async function saveGoal() {
    if (!goalConfirmed || goalSaving) return
    setGoalSaving(true)
    try {
      await api.updateGoal(sessionId, {
        target_column:    targetColumn,
        task_type:        taskType,
        confirmed_by_user: true
      })
      await refreshSession()
    } catch (err) {
      setUploadError("Could not save your selections. Please try again.")
    } finally {
      setGoalSaving(false)
    }
  }

  const needsPrivacy  = columnsToCheck.length > 0 && !privacyAck
  const allChosen     = columnsToCheck.every(c => privacyChoices[c])
  const goalSaved     = session?.goal?.confirmed_by_user === true
  const canContinue   = isComplete && !needsPrivacy && goalSaved

  if (uploading || stageRunning) {
    return (
      <div className="space-y-6">
        <StageHeader />
        <AgentRunning
          stageName={uploading ? "Uploading your file…" : "Reading and analysing your data…"}
          message="This usually takes a few seconds."
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StageHeader />

      {(uploadError || stageError) && (
        <AlertBanner
          type="error"
          title="Something went wrong"
          message={uploadError || stageError}
          onClose={() => setUploadError(null)}
        />
      )}

      {/* Upload drop zone */}
      {!isComplete && (
        <div
          className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer
                      transition-colors ${
                        dragOver
                          ? "border-blue-400 bg-blue-50"
                          : "border-gray-200 hover:border-blue-300 hover:bg-gray-50"
                      }`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          role="button"
          tabIndex={0}
          aria-label="Upload CSV file"
          onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
        >
          <Upload className="w-10 h-10 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-700 font-medium mb-1">Drop your CSV file here</p>
          <p className="text-gray-400 text-sm">or click to browse</p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
          />
        </div>
      )}

      {/* Results */}
      {isComplete && result && (
        <div className="space-y-5">
          {/* File summary */}
          <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-100 rounded-xl">
            <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
            <div>
              <p className="font-semibold text-green-800 text-sm">
                {result.file_name ?? "File loaded successfully"}
              </p>
              <p className="text-green-600 text-xs">
                {result.row_count?.toLocaleString()} rows &middot; {result.column_count} columns
              </p>
            </div>
          </div>

          {result.plain_english_summary && (
            <ExplanationPanel message={result.plain_english_summary} />
          )}

          {result.structural_check?.sample_rows && (
            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">Preview — first rows of your data</p>
              <DataPreviewTable
                rows={result.structural_check.sample_rows}
                columns={result.structural_check.column_names}
                maxRows={8}
              />
            </div>
          )}

          {/* Target column + task type selection */}
          {!goalSaved && columns.length > 0 && (
            <div className="border border-blue-200 bg-blue-50 rounded-2xl p-5 space-y-4">
              <p className="font-semibold text-blue-900">Tell us what you want to predict</p>
              <p className="text-blue-700 text-sm">
                Select the column your model should predict, and the type of problem.
              </p>

              <div className="space-y-3">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Which column do you want to predict?
                  </label>
                  <select
                    value={targetColumn}
                    onChange={e => setTargetColumn(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl border border-gray-200 bg-white text-sm text-gray-800
                               focus:outline-none focus:ring-2 focus:ring-blue-400"
                  >
                    <option value="">— Select a column —</option>
                    {columns.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    What kind of prediction is this?
                  </label>
                  <div className="flex gap-2">
                    {[
                      { id: "classification", label: "Categories",   desc: "e.g. yes/no, churn/stay" },
                      { id: "regression",     label: "Numbers",      desc: "e.g. price, revenue" },
                      { id: "time_series",    label: "Time Series",  desc: "e.g. forecast sales over time" }
                    ].map(({ id, label, desc }) => (
                      <button
                        key={id}
                        onClick={() => setTaskType(id)}
                        className={`flex-1 p-3 rounded-xl border text-left transition-colors ${
                          taskType === id
                            ? "bg-[#1B3A5C] text-white border-[#1B3A5C]"
                            : "bg-white border-gray-200 text-gray-700 hover:border-blue-300"
                        }`}
                      >
                        <p className="text-sm font-medium">{label}</p>
                        <p className={`text-xs mt-0.5 ${taskType === id ? "text-blue-200" : "text-gray-400"}`}>{desc}</p>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <button
                onClick={saveGoal}
                disabled={!goalConfirmed || goalSaving}
                className="w-full py-2.5 rounded-xl bg-[#1B3A5C] text-white text-sm font-medium
                           disabled:opacity-40 hover:bg-[#162f4d] transition-colors"
              >
                {goalSaving ? "Saving…" : "Confirm selections"}
              </button>
            </div>
          )}

          {goalSaved && (
            <div className="flex items-center gap-2 p-3 bg-green-50 border border-green-100 rounded-xl text-sm text-green-800">
              <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
              Predicting <strong className="mx-1">{session?.goal?.target_column}</strong>
              using <strong className="ml-1">{session?.goal?.task_type}</strong>
            </div>
          )}

          {/* Privacy gate */}
          {needsPrivacy && (
            <div className="border border-amber-200 bg-amber-50 rounded-2xl p-5 space-y-4">
              <div className="flex items-start gap-3">
                <Lock className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="font-semibold text-amber-800 mb-1">Sensitive columns detected</p>
                  <p className="text-amber-700 text-sm">
                    We found columns that may contain personal or sensitive information.
                    Please decide how each should be treated before we continue.
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                {columnsToCheck.map(col => (
                  <div key={col} className="bg-white rounded-xl border border-amber-100 p-4">
                    <p className="font-medium text-gray-800 text-sm mb-2">"{col}"</p>
                    <div className="flex gap-2 flex-wrap">
                      {[
                        { id: "include",   label: "Include as-is" },
                        { id: "anonymise", label: "Anonymise" },
                        { id: "exclude",   label: "Exclude" }
                      ].map(({ id, label }) => (
                        <button
                          key={id}
                          onClick={() => setPrivacyChoices(p => ({ ...p, [col]: id }))}
                          className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                            privacyChoices[col] === id
                              ? "bg-[#1B3A5C] text-white border-[#1B3A5C]"
                              : "border-gray-200 text-gray-600 hover:border-blue-300"
                          }`}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                    {!privacyChoices[col] && (
                      <p className="text-xs text-amber-600 mt-1.5">Choose an option above</p>
                    )}
                  </div>
                ))}
              </div>

              <button
                onClick={submitPrivacy}
                disabled={privacySubmitting || !allChosen}
                className="w-full py-2.5 rounded-xl bg-[#1B3A5C] text-white text-sm font-medium
                           disabled:opacity-40 hover:bg-[#162f4d] transition-colors"
              >
                {privacySubmitting ? "Saving…" : "Confirm privacy decisions"}
              </button>
            </div>
          )}

          {privacyAck && columnsToCheck.length > 0 && (
            <AlertBanner
              type="success"
              title="Privacy decisions saved"
              message="Your data privacy preferences have been recorded and will be applied throughout the pipeline."
            />
          )}
        </div>
      )}

      <StageNavigation
        onContinue={goToNextStage}
        continueDisabled={!canContinue}
        continueLabel="Continue to Quality Check"
      />
    </div>
  )
}

function StageHeader() {
  return (
    <div>
      <h2 className="text-2xl font-serif text-gray-900 mb-1">Load Your Data</h2>
      <p className="text-gray-500 text-sm">
        Upload the CSV file you want to build your model from.
        We'll check the file for any issues and show you a preview.
      </p>
    </div>
  )
}
