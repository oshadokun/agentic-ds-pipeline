import { useState } from "react"
import Header from "./Header"
import ProgressSidebar from "./ProgressSidebar"
import ConfirmModal from "../shared/ConfirmModal"
import { useSession } from "../../contexts/SessionContext"
import { usePipeline } from "../../contexts/PipelineContext"

/**
 * AppShell
 * The outer frame of the application when a session is active.
 * Renders the header, left sidebar, and the main content area.
 * The children prop receives the current stage view.
 */
export default function AppShell({ children }) {
  const { session, deleteSession }  = useSession()
  const { currentStage }            = usePipeline()
  const [confirmDelete, setConfirmDelete] = useState(false)

  async function handleDelete() {
    setConfirmDelete(false)
    await deleteSession(session.session_id)
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-gray-50">
      {/* Top header bar */}
      <Header />

      {/* Body: sidebar + main content */}
      <div className="flex flex-1 min-h-0">

        {/* Progress sidebar — hidden on small screens */}
        <div className="hidden md:flex">
          <ProgressSidebar onDeleteSession={() => setConfirmDelete(true)} />
        </div>

        {/* Main content area */}
        <main
          className="flex-1 overflow-y-auto"
          id="main-content"
          aria-label="Pipeline stage content"
        >
          <div className="max-w-3xl mx-auto px-6 py-8">
            {children}
          </div>
        </main>
      </div>

      {/* Delete confirmation modal */}
      {confirmDelete && (
        <ConfirmModal
          title="Delete this session?"
          message="All data, models, and reports for this session will be permanently removed."
          consequence="This cannot be undone. You will need to start a new pipeline from scratch."
          confirmLabel="Delete Session"
          onConfirm={handleDelete}
          onCancel={() => setConfirmDelete(false)}
        />
      )}
    </div>
  )
}
